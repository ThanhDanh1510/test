import os
import pandas as pd
import numpy as np
import ast

class MedPRSDatasetLoader:
    def __init__(self, data_dir="./"):
        """
        data_dir: Path to dataset folder.
        On Kaggle, default is usually '/kaggle/input/medprs-dataset/'
        """
        self.data_dir = data_dir
        self.journal_df = None
        self.label_to_journal = {}
        self.journal_to_label = {}
        
    def find_file(self, filename):
        """Helper to locate file in current dir or kaggle input dir"""
        candidates = [
            os.path.join(self.data_dir, filename),
            os.path.join("/kaggle/input/medprs-dataset/", filename),
            os.path.join("/kaggle/input/medprs-dataset/", filename.replace(".csv", "")),
            os.path.join("./", filename)
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def load_journals(self, journal_csv_name="journal_full_info.csv"):
        """Loads and prepares journal metadata from journal_full_info.csv"""
        file_path = self.find_file(journal_csv_name)
        if not file_path:
            # Fallback to journal_category.csv if available
            file_path = self.find_file("journal_category.csv")
            
        if not file_path:
            raise FileNotFoundError(f"Could not find journal metadata file ({journal_csv_name} or journal_category.csv).")

        print(f"[DatasetLoader] Loading journal metadata from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Normalize column names & parse types
        processed_journals = []
        for idx, row in df.iterrows():
            # Handle Label
            label_val = row.get('Label', idx)
            try:
                label_id = int(label_val)
            except ValueError:
                label_id = idx

            # Handle Categories
            raw_cats = str(row.get('Best Categories', row.get('Categories', '[]')))
            try:
                cats_cleaned = raw_cats.replace(';', ',')
                categories = ast.literal_eval(cats_cleaned) if cats_cleaned.startswith('[') else [cats_cleaned]
            except Exception:
                categories = [raw_cats]

            # Domain Flags (9 columns)
            domain_cols = [
                'Medicine', 'Neuroscience', 'Immunology and Microbiology',
                'Biochemistry, Genetics and Molecular Biology',
                'Pharmacology, Toxicology and Pharmaceutics',
                'Health Professions', 'Nursing', 'Psychology', 'Dentistry'
            ]
            domain_flags = {}
            for col in domain_cols:
                domain_flags[col] = float(row.get(col, 0.0))

            journal_obj = {
                "journal_id": label_id,
                "title": str(row.get('Journal', row.get('title', ''))).strip(),
                "aims": str(row.get('Aims', '')).strip(),
                "scope": str(row.get('Scope', '')).strip(),
                "categories": categories,
                "sjr_index": float(str(row.get('SJR index', 0)).replace(',', '.')) if pd.notnull(row.get('SJR index')) else 0.0,
                "best_quartile": str(row.get('Best Quartile', 'Q4')).strip(),
                "h_index": int(row.get('H index', 0)) if pd.notnull(row.get('H index')) and str(row.get('H index')).isdigit() else 0,
                "domain_flags": domain_flags,
                "pubmed_url": str(row.get('URL', '')),
                "scimago_url": str(row.get('URL_Scimago', ''))
            }

            processed_journals.append(journal_obj)
            self.label_to_journal[label_id] = journal_obj
            self.journal_to_label[journal_obj["title"].lower()] = label_id

        self.journal_df = pd.DataFrame(processed_journals)
        print(f"[DatasetLoader] Successfully loaded {len(self.journal_df)} journals.")
        return self.journal_df

    def load_papers(self, split="test", nrows=None):
        """Loads test_set.csv, val_set.csv, or train_set.csv"""
        filename = f"{split}_set.csv"
        file_path = self.find_file(filename)
        if not file_path:
            raise FileNotFoundError(f"Could not locate {filename}.")

        print(f"[DatasetLoader] Loading paper dataset ({split}) from: {file_path}")
        df = pd.read_csv(file_path, nrows=nrows)
        
        # Clean null values
        df['Title'] = df['Title'].fillna('').astype(str)
        df['Abstract'] = df['Abstract'].fillna('').astype(str)
        df['Keywords'] = df['Keywords'].fillna('').astype(str)
        if 'Label' in df.columns:
            df['Label'] = df['Label'].astype(int)

        print(f"[DatasetLoader] Loaded {len(df)} papers for split='{split}'.")
        return df

if __name__ == "__main__":
    loader = MedPRSDatasetLoader(data_dir="./")
    journals = loader.load_journals()
    print("Sample Journal:", journals.iloc[0]["title"], "| Quartile:", journals.iloc[0]["best_quartile"])
