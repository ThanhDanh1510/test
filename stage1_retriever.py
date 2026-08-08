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
    BioBERT SimCPSR Dual-Branch Architecture (Approach C in MedPRS)
    - Projection Layer (Linear + ReLU + Dropout)
    - Cosine Similarity & Classifier Layer
    """
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", projection_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Sequential(
            nn.Linear(768, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Pooler output or CLS token
        cls_rep = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_rep)
        # Normalize for Cosine Similarity Vector Search
        normed = nn.functional.normalize(projected, p=2, dim=1)
        return normed

class Stage1Retriever:
    def __init__(self, journal_df, checkpoint_path=None, device=None):
        self.journal_df = journal_df
        self.journal_ids = []
        
        if HAS_TORCH:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[Stage1Retriever] Initializing encoder on device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.model = BioBERTSimCPSREncoder().to(self.device)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load_simcprs_weights(checkpoint_path)
            else:
                print(f"[Stage1Retriever] Note: No checkpoint_path passed or found. Using base BioBERT weights.")

            self.model.eval()
            self._build_journal_faiss_index()

    def _load_simcprs_weights(self, checkpoint_path):
        print(f"[Stage1Retriever] Loading fine-tuned SimCPSR checkpoint: {checkpoint_path}")
        try:
            w_path = checkpoint_path
            if os.path.isdir(checkpoint_path):
                # If directory contains PyTorch state files directly (like data.pkl)
                data_pkl = os.path.join(checkpoint_path, "data.pkl")
                if os.path.exists(data_pkl):
                    w_path = data_pkl
                else:
                    # Look for subfolders or files inside
                    for root, dirs, files in os.walk(checkpoint_path):
                        for d in dirs:
                            sub_pkl = os.path.join(root, d, "data.pkl")
                            if os.path.exists(sub_pkl):
                                w_path = sub_pkl
                                break
                        if w_path != checkpoint_path:
                            break
                        for f in files:
                            if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".bin") or "simcprs" in f.lower() or "epoch" in f.lower():
                                w_path = os.path.join(root, f)
                                break
                        if w_path != checkpoint_path:
                            break

            print(f"[Stage1Retriever] Loading weights file: {w_path}")
            raw_state = torch.load(w_path, map_location=self.device)
            if isinstance(raw_state, dict):
                if 'model_state_dict' in raw_state:
                    raw_state = raw_state['model_state_dict']
                elif 'state_dict' in raw_state:
                    raw_state = raw_state['state_dict']
                elif 'model' in raw_state:
                    raw_state = raw_state['model']

            model_keys = set(self.model.state_dict().keys())
            cleaned_state = {}

            for k, v in raw_state.items():
                new_k = k
                # Strip common prefixes from DataParallel / Peft / Base_model checkpoints
                for prefix in [
                    'module.base_model.', 'base_model.',
                    'module.paper_encoder.', 'paper_encoder.',
                    'module.paper_branch.', 'paper_branch.',
                    'module.', 'model.'
                ]:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]

                if new_k in model_keys:
                    cleaned_state[new_k] = v
                elif f"encoder.{new_k}" in model_keys:
                    cleaned_state[f"encoder.{new_k}"] = v
                elif new_k.startswith("bert.") and f"encoder.{new_k[5:]}" in model_keys:
                    cleaned_state[f"encoder.{new_k[5:]}"] = v
                elif f"projection.{new_k}" in model_keys:
                    cleaned_state[f"projection.{new_k}"] = v

            matched_count = len(cleaned_state)
            print(f"[Stage1Retriever] Key matching report: Matched {matched_count}/{len(model_keys)} layers.")
            
            if matched_count > 0:
                self.model.load_state_dict(cleaned_state, strict=False)
                print("[Stage1Retriever] Successfully loaded fine-tuned SimCPSR weights into BioBERT!")
            else:
                print(f"[Stage1Retriever] Warning: 0 keys matched. Sample checkpoint keys: {list(raw_state.keys())[:5]}")
        except Exception as e:
            print(f"[Stage1Retriever] Error loading checkpoint ({checkpoint_path}): {e}")

    def _build_journal_faiss_index(self):
        """Encodes all 1,408 journals into FAISS IndexFlatIP (Cosine similarity)"""
        print("[Stage1Retriever] Pre-computing journal embeddings for FAISS Index...")
        embeddings = []
        self.journal_ids = []

        with torch.no_grad():
            for idx, row in self.journal_df.iterrows():
                cats = row['categories']
                if isinstance(cats, list):
                    cats_flat = []
                    for c in cats:
                        if isinstance(c, list):
                            cats_flat.extend([str(x) for x in c])
                        else:
                            cats_flat.append(str(c))
                    cats_str = ', '.join(cats_flat)
                else:
                    cats_str = str(cats)

                j_text = f"Journal: {row['title']}. Aims: {row['aims']}. Scope: {row['scope']}. Categories: {cats_str}"
                inputs = self.tokenizer(j_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
                emb = self.model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()[0]
                embeddings.append(emb)
                self.journal_ids.append(row['journal_id'])

        emb_matrix = np.array(embeddings).astype('float32')
        self.journal_embeddings = emb_matrix
        
        if HAS_FAISS:
            dim = emb_matrix.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(emb_matrix)
            print(f"[Stage1Retriever] Successfully indexed {self.index.ntotal} journal vectors in FAISS.")

    def retrieve(self, paper_object, top_k=50):
        p_text = f"Title: {paper_object['title']}. Abstract: {paper_object['abstract']}. Keywords: {paper_object['keywords']}"
        
        if HAS_TORCH:
            inputs = self.tokenizer(p_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                query_emb = self.model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy().astype('float32')
            if HAS_FAISS and hasattr(self, 'index') and self.index is not None:
                scores, indices = self.index.search(query_emb, top_k)
                top_scores = scores[0]
                top_indices = indices[0]
            else:
                sims = np.dot(self.journal_embeddings, query_emb[0])
                top_indices = np.argsort(sims)[::-1][:top_k]
                top_scores = sims[top_indices]
        else:
            query_emb = self.vectorizer.transform([p_text]).toarray()[0]
            sims = np.dot(self.journal_embeddings, query_emb)
            top_indices = np.argsort(sims)[::-1][:top_k]
            top_scores = sims[top_indices]
        
        results = []
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
            j_id = self.journal_ids[idx]
            j_row = self.journal_df[self.journal_df['journal_id'] == j_id].iloc[0].to_dict()
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
