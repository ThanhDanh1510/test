import os
import json
import time

from dataset_loader import MedPRSDatasetLoader
from stage0_parser import Stage0Parser
from stage1_retriever import Stage1Retriever
from stage2_hybrid_gate import Stage2HybridGate
from stage3_dear_reranker import Stage3DeARReranker
from evaluator import evaluate_pipeline

class DTARSlimPipeline:
    """
    Master DTAR-Slim v2.1 Pipeline Orchestrator
    Stage 0: Fast Parser (PICO, Study Type, Soft Domain Scores)
    Stage 1: BioBERT SimCPSR Vector Search (FAISS Index)
    Stage 2: Python Hybrid Gate (Integrity Check + SetFit Desk Reject + Soft Domain Match)
    Stage 3: DeAR Reranker (Pointwise -> Listwise + Dynamic Kendall Tau Permutation + Calibrated Faithfulness Check)
    """
    def __init__(self, data_dir="./", checkpoint_path=None):
        print("\n=======================================================")
        print("Initializing DTAR-Slim v2.1 Pipeline...")
        print("=======================================================\n")
        
        self.loader = MedPRSDatasetLoader(data_dir=data_dir)
        self.journal_df = self.loader.load_journals()
        
        if not checkpoint_path:
            checkpoint_path = self._auto_find_checkpoint(data_dir)

        self.stage0 = Stage0Parser()
        self.stage1 = Stage1Retriever(self.journal_df, checkpoint_path=checkpoint_path)
        self.stage2 = Stage2HybridGate()
        self.stage3 = Stage3DeARReranker()

    def _auto_find_checkpoint(self, data_dir):
        """Auto-detects SimCPSR checkpoint folder or file"""
        search_dirs = [data_dir, "./", "/kaggle/input/"]
        for s_dir in search_dirs:
            if s_dir and os.path.exists(s_dir):
                for root, dirs, files in os.walk(s_dir):
                    for d in dirs:
                        if "simcprs" in d.lower() or "epoch" in d.lower():
                            ckpt = os.path.join(root, d)
                            print(f"[DTARSlimPipeline] Auto-detected SimCPSR checkpoint folder: {ckpt}")
                            return ckpt
                    for f in files:
                        if f.endswith(".pt") or f.endswith(".bin") or "simcprs" in f.lower():
                            ckpt = os.path.join(root, f)
                            print(f"[DTARSlimPipeline] Auto-detected SimCPSR checkpoint file: {ckpt}")
                            return ckpt
        return None

    def run(self, paper_input, user_strict_mode=False, return_stage1=False):
        """
        Executes full End-to-End Pipeline for a single paper query.
        """
        # Stage 0: Parse Paper
        paper_obj = self.stage0.parse_paper(
            title=paper_input.get('title', ''),
            abstract=paper_input.get('abstract', ''),
            keywords=paper_input.get('keywords', '')
        )

        # Stage 1: BioBERT SimCPSR Dense Retrieval -> Top 50
        top_50_candidates = self.stage1.retrieve(paper_obj, top_k=50)

        # Stage 2: Python Hybrid Gate -> Top 15 - 20 Candidates
        pruned_candidates = self.stage2.process_candidates(top_50_candidates, paper_obj, user_strict_mode=user_strict_mode)

        # Stage 3: DeAR Dual-Stage Reranking & Reasoning Trace -> Top 5
        final_recommendations = self.stage3.rerank_and_explain(paper_obj, pruned_candidates)

        if return_stage1:
            return final_recommendations, top_50_candidates
        return final_recommendations

def main():
    # 1. Initialize Pipeline
    pipeline = DTARSlimPipeline(data_dir="./")

    # 2. Run Demo Single Query
    sample_paper = {
        "title": "A chemo mechanical constitutive model for muscle activation in bat wing skins.",
        "abstract": "Birds, bats and insects have evolved unique wing structures to achieve a wide range of flight capabilities. Insects have relatively stiff and passive wings...",
        "keywords": "bat wing skin; chemo mechanical; constitutive modelling; skeletal muscle"
    }

    print("\n[Demo] Running Single Query Recommendation...")
    recommendations = pipeline.run(sample_paper)
    print(json.dumps(recommendations[0], indent=2, ensure_ascii=False))

    # 3. Optional Benchmark Run on Test Set
    try:
        test_df = pipeline.loader.load_papers(split="test", nrows=50)
        evaluate_pipeline(pipeline, test_df)
    except Exception as e:
        print(f"\n[Note] Skipping test_set evaluation: {e}")

if __name__ == "__main__":
    main()
