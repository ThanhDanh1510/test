import os
import numpy as np
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[Stage1Retriever] 'transformers' or 'torch' module not found. Falling back to Lightweight TF-IDF / Feature Vector Search.")

class BioBERTSimCPSREncoder(nn.Module):
    """
    SimCPSRModel (Exact MedPRS Dual-Branch Classifier Architecture from tnt1626/DeAR-Reranking)
    - Mean pooling across BioBERT token embeddings
    - linear1_1 (768 -> 512) for Paper Projection
    - linear2_1 (768 -> 512) for Journal Projection
    - linear_main_1 (1918 -> 1406) for Joint Similarity Logits
    """
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", num_classes=1406):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.linear1_1 = nn.Linear(768, 512)
        self.linear2_1 = nn.Linear(768, 512)
        self.linear_main_1 = nn.Linear(1918, num_classes)

    def mean_pooling(self, outputs, attention_mask):
        token_embeddings = outputs[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        emb = torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        return emb

    def encode_paper(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.mean_pooling(outputs, attention_mask)
        paper_proj = torch.relu(self.linear1_1(emb))
        return paper_proj

    def encode_journal(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.mean_pooling(outputs, attention_mask)
        journal_proj = torch.relu(self.linear2_1(emb))
        return journal_proj

    def forward_logits(self, paper_proj, journal_proj_embeddings):
        paper_proj_norm = paper_proj / torch.clamp(paper_proj.norm(dim=-1, keepdim=True), min=1e-9)
        journal_proj_norm = journal_proj_embeddings / torch.clamp(journal_proj_embeddings.norm(dim=-1, keepdim=True), min=1e-9)
        
        sim_vector = torch.matmul(paper_proj_norm, journal_proj_norm.t())
        joint = torch.cat([paper_proj, sim_vector], dim=-1)
        logits = self.linear_main_1(joint)
        return logits

class Stage1Retriever:
    def __init__(self, journal_df, checkpoint_path=None, device=None):
        self.journal_df = journal_df
        self.journal_ids = self.journal_df['journal_id'].tolist()
        num_classes = len(self.journal_df)
        
        if HAS_TORCH:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[Stage1Retriever] Initializing encoder on device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.model = BioBERTSimCPSREncoder(num_classes=num_classes).to(self.device)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load_simcprs_weights(checkpoint_path)
            else:
                print(f"[Stage1Retriever] Note: No checkpoint_path passed or found. Using base BioBERT weights.")

            self.model.eval()
            self._precompute_journal_embeddings()

    def _load_simcprs_weights(self, checkpoint_path):
        print(f"[Stage1Retriever] Loading fine-tuned SimCPSR checkpoint: {checkpoint_path}")
        w_path = checkpoint_path
        if os.path.isdir(checkpoint_path):
            data_pkl = os.path.join(checkpoint_path, "data.pkl")
            if not os.path.exists(data_pkl):
                for root, dirs, files in os.walk(checkpoint_path):
                    if "data.pkl" in files:
                        w_path = root
                        break
            if os.path.exists(os.path.join(w_path, "data.pkl")):
                import zipfile, tempfile
                print(f"[Stage1Retriever] Re-packing PyTorch 2.x unzipped directory '{w_path}' into valid PyTorch archive...")
                temp_zip_path = os.path.join(tempfile.gettempdir(), "temp_simcprs_ckpt.pt")
                archive_name = os.path.basename(w_path.rstrip("/\\")) or "archive"
                with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_STORED) as zip_f:
                    for root, dirs, files in os.walk(w_path):
                        for file in files:
                            abs_path = os.path.join(root, file)
                            rel_path = os.path.relpath(abs_path, w_path)
                            zip_f.write(abs_path, os.path.join(archive_name, rel_path))
                w_path = temp_zip_path

        try:
            try:
                ckpt = torch.load(w_path, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(w_path, map_location=self.device)

            if isinstance(w_path, str) and w_path.endswith("temp_simcprs_ckpt.pt") and os.path.exists(w_path):
                try: os.remove(w_path)
                except: pass

            raw_state = ckpt
            if isinstance(ckpt, dict):
                if 'model_state_dict' in ckpt:
                    raw_state = dict(ckpt['model_state_dict'])
                    if 'classifier_head' in ckpt and isinstance(ckpt['classifier_head'], dict):
                        for k, v in ckpt['classifier_head'].items():
                            raw_state[k] = v
                elif 'state_dict' in ckpt:
                    raw_state = dict(ckpt['state_dict'])

            model_keys = set(self.model.state_dict().keys())
            cleaned_state = {}

            for k, v in raw_state.items():
                new_k = k
                for prefix in [
                    'module.base_model.', 'base_model.',
                    'module.paper_encoder.', 'paper_encoder.',
                    'module.paper_branch.', 'paper_branch.',
                    'module.', 'model.'
                ]:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                        break

                if new_k in model_keys:
                    cleaned_state[new_k] = v
                elif f"encoder.{new_k}" in model_keys:
                    cleaned_state[f"encoder.{new_k}"] = v
                elif new_k.startswith("bert.") and f"encoder.{new_k[5:]}" in model_keys:
                    cleaned_state[f"encoder.{new_k[5:]}"] = v
                elif f"projection.{new_k}" in model_keys:
                    cleaned_state[f"projection.{new_k}"] = v

            matched_count = len(cleaned_state)
            has_linear1 = "linear1_1.weight" in cleaned_state
            has_linear_main = "linear_main_1.weight" in cleaned_state
            print(f"[Stage1Retriever] Key matching report: Matched {matched_count}/{len(model_keys)} layers (Linear Heads Loaded: linear1_1={has_linear1}, linear_main_1={has_linear_main}).")
            
            if matched_count > 0:
                self.model.load_state_dict(cleaned_state, strict=False)
                print("[Stage1Retriever] Successfully loaded fine-tuned SimCPSR weights into BioBERT!")
            else:
                print(f"[Stage1Retriever] Warning: 0 keys matched. Sample checkpoint keys: {list(raw_state.keys())[:5]}")
        except Exception as e:
            print(f"[Stage1Retriever] Error loading checkpoint ({checkpoint_path}): {e}")

    def _precompute_journal_embeddings(self):
        """Encodes all 1,408 journals into 512-dim journal projected embeddings using linear2_1"""
        print("[Stage1Retriever] Pre-computing 512-dim journal projected embeddings with linear2_1...")
        j_embeddings = []
        with torch.no_grad():
            for idx, row in self.journal_df.iterrows():
                j_title = str(row['title']).strip()
                inputs = self.tokenizer(j_title, max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
                j_proj = self.model.encode_journal(inputs['input_ids'], inputs['attention_mask'])
                j_embeddings.append(j_proj.squeeze(0))

        self.journal_proj_tensor = torch.stack(j_embeddings).to(self.device)
        print(f"[Stage1Retriever] Successfully pre-computed {self.journal_proj_tensor.size(0)} journal projected embeddings tensor of shape {self.journal_proj_tensor.shape}.")

    def retrieve(self, paper_input, abstract=None, top_k=50):
        """
        Retrieves Top K candidate journals.
        Accepts either paper_dict or (title, abstract, top_k).
        """
        if isinstance(paper_input, dict):
            title = paper_input.get('title', '')
            abstract_text = paper_input.get('abstract', '')
            p_text = f"{title} {abstract_text}".strip()
            if not p_text:
                p_text = paper_input.get('keywords', 'Medical research paper')
        else:
            title = str(paper_input)
            abstract_text = str(abstract) if abstract else ""
            p_text = f"{title} {abstract_text}".strip()

        if HAS_TORCH:
            inputs = self.tokenizer(p_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                paper_proj = self.model.encode_paper(inputs['input_ids'], inputs['attention_mask'])
                logits = self.model.forward_logits(paper_proj, self.journal_proj_tensor)
                top_scores, top_indices = torch.topk(logits[0], k=min(top_k, logits.size(1)))
                top_scores = top_scores.cpu().numpy()
                top_indices = top_indices.cpu().numpy()
        else:
            query_emb = self.vectorizer.transform([p_text]).toarray()[0]
            sims = np.dot(self.journal_embeddings, query_emb)
            top_indices = np.argsort(sims)[::-1][:top_k]
            top_scores = sims[top_indices]
        
        results = []
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
            if idx < len(self.journal_df):
                j_row = self.journal_df.iloc[idx].to_dict()
            else:
                j_row = self.journal_df.iloc[idx % len(self.journal_df)].to_dict()
            j_row['dense_similarity_score'] = float(score)
            results.append(j_row)

        return results

if __name__ == "__main__":
    from dataset_loader import MedPRSDatasetLoader
    loader = MedPRSDatasetLoader()
    journals = loader.load_journals()
    
    retriever = Stage1Retriever(journals)
    sample_paper = {
        "title": "Genetics of migraine: where are we now?",
        "abstract": "Migraine is a complex brain disorder...",
        "keywords": "Migraine; Genetics"
    }
    top_50 = retriever.retrieve(sample_paper, top_k=50)
    print("Retrieved Top 1 Journal:", top_50[0]['title'], "| Sim Score:", top_50[0]['dense_similarity_score'])
