import os
import json
import time
from dataset_loader import MedPRSDatasetLoader
from stage0_parser import Stage0Parser
from stage0_5_policy_encoder import JournalPolicyEncoder
from stage1_retriever import Stage1Retriever
from stage2_hybrid_gate import Stage2RiskGate
from stage3_strategic_scorer import Stage3StrategicScorer
from stage5_pareto_recommender import Stage5ParetoRecommender
from stage6_uncertainty import Stage6UncertaintyLayer
from stage7_evidence_explainer import Stage7EvidenceExplainer
from evaluator import evaluate_pipeline

class DTARv3Pipeline:
    """
    DTAR-Submission Strategist v3.0 Master Pipeline:
    Risk-Aware, Counterfactual, Pareto and Uncertainty-Calibrated Medical Journal Recommendation
    """
    def __init__(self, data_dir="./", checkpoint_path=None):
        print("=======================================================")
        print("INITIALIZING DTAR-SUBMISSION STRATEGIST v3.0 PIPELINE")
        print(f"Data Directory: {data_dir}")
        print("=======================================================")

        self.loader = MedPRSDatasetLoader(data_dir=data_dir)
        self.journal_df = self.loader.load_journals()
        
        # 1. Pipeline Stages
        self.stage0 = Stage0Parser()
        self.policy_encoder = JournalPolicyEncoder()
        
        # Auto-detect checkpoint
        actual_ckpt = checkpoint_path or self._auto_find_checkpoint(data_dir)
        self.stage1 = Stage1Retriever(self.journal_df, checkpoint_path=actual_ckpt)
        self.stage2 = Stage2RiskGate()
        self.stage3 = Stage3StrategicScorer()
        self.stage5 = Stage5ParetoRecommender()
        self.stage6 = Stage6UncertaintyLayer(coverage_target=0.90)
        self.stage7 = Stage7EvidenceExplainer()

        print("✅ DTAR v3.0 Pipeline successfully loaded and ready for inference!\n")

    def _auto_find_checkpoint(self, data_dir):
        candidates = [
            "/kaggle/input/datasets/tintngc/medprs-dataset/Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth",
            "/kaggle/working/best_simcprs_checkpoint.pth",
            "/kaggle/working/latest_step_checkpoint.pth",
            os.path.join(data_dir, "Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth")
        ]
        for c in candidates:
            if os.path.exists(c):
                print(f"[Pipeline] Auto-detected Checkpoint: {c}")
                return c
        if os.path.exists("/kaggle/input/"):
            for root, dirs, files in os.walk("/kaggle/input/"):
                for f in files:
                    if "epoch_02" in f.lower() or "epoch02" in f.lower():
                        detected = os.path.join(root, f)
                        print(f"[Pipeline] Auto-detected Checkpoint in Kaggle Input: {detected}")
                        return detected
        return None

    def run(self, paper_dict, user_preferences=None, return_stage1=False):
        """
        Executes full DTAR v3.0 End-to-End Recommendation:
        Stage 0 -> Stage 1 -> Stage 2 -> Stage 3/4 -> Stage 5 -> Stage 6 -> Stage 7
        """
        title = paper_dict.get('title', '')
        abstract = paper_dict.get('abstract', '')
        keywords = paper_dict.get('keywords', '')

        # STAGE 0: Structured Understanding
        parsed_paper = self.stage0.parse_paper(title, abstract, keywords)

        # STAGE 1: Dense Retrieval (Top 50 Candidates)
        stage1_top50 = self.stage1.retrieve(title, abstract, top_k=50)

        # STAGE 2: Risk-Aware Policy Gate (Top 20 Candidates)
        stage2_top20 = self.stage2.process_candidates(stage1_top50, parsed_paper)

        # STAGE 3 & 4: Strategic Utility Scoring
        scored_candidates = self.stage3.score_candidates(parsed_paper, stage2_top20, user_preferences)

        # STAGE 5: Pareto Frontier Recommendation
        pareto_candidates = self.stage5.compute_pareto_frontier(scored_candidates)

        # STAGE 6: Uncertainty Layer & 90% Conformal Confidence Set
        calibrated_candidates, conf_set = self.stage6.estimate_uncertainty(pareto_candidates)

        # STAGE 7: Evidence-Grounded Explanation (Top 5 Results)
        top5_recommendations = self.stage7.explain_recommendations(parsed_paper, calibrated_candidates)

        final_response = {
            "paper_summary": {
                "study_type": parsed_paper['study_type'],
                "pico": parsed_paper['pico'],
                "paper_signals": parsed_paper['paper_signals']
            },
            "recommendations": top5_recommendations,
            "confidence_set": conf_set,
            "system_notes": {
                "pipeline_version": "DTAR-Submission Strategist v3.0",
                "total_candidates_evaluated": len(self.journal_df),
                "retrieval_pool_size": len(stage1_top50),
                "pareto_options_count": sum(1 for c in pareto_candidates if c.get('pareto_optimal', False))
            }
        }

        if return_stage1:
            return top5_recommendations, stage1_top50

        return final_response

# Aliases for backward compatibility
DTARSlimPipeline = DTARv3Pipeline

def main():
    pipeline = DTARv3Pipeline(data_dir="./")
    sample_paper = {
        "title": "A chemo mechanical constitutive model for muscle activation in bat wing skins.",
        "abstract": "Birds, bats and insects have evolved unique wing structures to achieve a wide range of flight capabilities...",
        "keywords": "bat wing skin; chemo mechanical; constitutive modelling; skeletal muscle"
    }
    result = pipeline.run(sample_paper)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
