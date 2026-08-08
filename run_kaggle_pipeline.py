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
        """Auto-detects SimCPSR checkpoint folder or file (prioritizing Epoch3 / newest checkpoints)"""
        search_dirs = [data_dir, "./", "/kaggle/input/"]
        found_ckpts = []
        for s_dir in search_dirs:
            if s_dir and os.path.exists(s_dir):
                for root, dirs, files in os.walk(s_dir):
                    for d in dirs:
                        d_lower = d.lower()
                        if "epoch3" in d_lower or "latest_step" in d_lower or "epoch_03" in d_lower:
                            ckpt = os.path.join(root, d)
                            print(f"[DTARSlimPipeline] Found Epoch3/Latest Checkpoint Folder: {ckpt}")
                            return ckpt
                        elif "simcprs" in d_lower or "epoch" in d_lower:
                            found_ckpts.append(os.path.join(root, d))
                    for f in files:
                        f_lower = f.lower()
                        if "epoch3" in f_lower or "latest_step" in f_lower:
                            ckpt = os.path.join(root, f)
                            print(f"[DTARSlimPipeline] Found Epoch3/Latest Checkpoint File: {ckpt}")
                            return ckpt
                        elif f.endswith(".pt") or f.endswith(".bin") or f.endswith(".pth") or "simcprs" in f_lower:
                            found_ckpts.append(os.path.join(root, f))

        if found_ckpts:
            found_ckpts.sort(key=lambda x: ("epoch3" in x.lower() or "latest" in x.lower() or "epoch_03" in x.lower()), reverse=True)
            print(f"[DTARSlimPipeline] Auto-detected checkpoint: {found_ckpts[0]}")
            return found_ckpts[0]
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

def run_master_kaggle_workflow(data_dir="/kaggle/input/medprs-dataset/", output_dir="/kaggle/working/", epochs=10, test_samples=1000):
    """
    Master Continuous Kaggle Workflow:
    1. Runs Overnight Training (BioBERT SimCPSR 10 Epochs, 2x T4 GPU, FP16, Step Checkpoint)
    2. Auto-loads the newly trained best_simcprs_checkpoint.pth
    3. Runs Single Query Demo Recommendation
    4. Runs Batch Evaluation on test_set.csv and prints final report
    """
    print("\n=======================================================")
    print("PHASE 1: RUNNING OVERNIGHT TRAINING (10 EPOCHS)")
    print("=======================================================\n")
    
    from train_simcprs import train_overnight
    train_overnight(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=64,
        lr=5e-5,
        save_step_frequency=1000,
        use_fp16=True,
        resume_training=True
    )

    print("\n=======================================================")
    print("PHASE 2: LOADING NEWLY TRAINED BEST CHECKPOINT FOR EVALUATION")
    print("=======================================================\n")
    
    best_ckpt = os.path.join(output_dir, "best_simcprs_checkpoint.pth")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(output_dir, "latest_step_checkpoint.pth")

    pipeline = DTARSlimPipeline(data_dir=data_dir, checkpoint_path=best_ckpt)

    print("\n=======================================================")
    print("PHASE 3: RUNNING SINGLE QUERY DEMO RECOMMENDATION")
    print("=======================================================\n")
    
    sample_paper = {
        "title": "Genetics of migraine: where are we now?",
        "abstract": "Migraine is a complex brain disorder explained by genetic and environmental factors. Familial hemiplegic migraine is a rare monogenic subtype...",
        "keywords": "Migraine; Genetics; Genome wide association studies"
    }

    recommendations = pipeline.run(sample_paper)
    print(json.dumps(recommendations[0], indent=2, ensure_ascii=False))

    print(f"\n=======================================================")
    print(f"PHASE 4: RUNNING BATCH EVALUATION ON TEST SET ({test_samples} SAMPLES)")
    print("=======================================================\n")

    try:
        test_df = pipeline.loader.load_papers(split="test", nrows=test_samples)
        evaluate_pipeline(pipeline, test_df)
    except Exception as e:
        print(f"⚠️ Batch evaluation note: {e}")

def main():
    pipeline = DTARSlimPipeline(data_dir="./")
    sample_paper = {
        "title": "A chemo mechanical constitutive model for muscle activation in bat wing skins.",
        "abstract": "Birds, bats and insects have evolved unique wing structures to achieve a wide range of flight capabilities...",
        "keywords": "bat wing skin; chemo mechanical; constitutive modelling; skeletal muscle"
    }
    recommendations = pipeline.run(sample_paper)
    print(json.dumps(recommendations[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
