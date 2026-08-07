# DTAR-Slim v2.1: Medical Paper-to-Journal Recommendation System

**DTAR-Slim v2.1** is a high-speed, highly interpretable medical paper submission recommendation pipeline designed for biomedical research papers (Title, Abstract, Keywords) matching against PubMed/SCImago indexed journals (`journal_full_info.csv`).

---

## 🚀 Key Features

1. **High-Speed Dense Retrieval (< 10ms)**: BioBERT SimCPSR FAISS Vector Search narrowing 1,408 PubMed journals down to Top-50 candidates.
2. **Python Hybrid Gate (0ms LLM)**: Hard Integrity Gate (DOAJ / Beall's List / Delisted check), SetFit Desk Reject Engine (Case Report, Cell Line exclusions), and Soft Domain Scoring.
3. **DeAR Dual-Stage Reranking**: Pointwise Scorer + Single-Pass Listwise CoT Reranker with **Scope-Over-Prestige Prompting**.
4. **Dynamic Adaptive Permutation Check**: Evaluates position bias using Kendall's $\tau$ correlation across input permutations.
5. **Calibrated Faithfulness Check**: F1-calibrated similarity check preventing LLM hallucination in generated reasoning traces.
6. **Kaggle Notebook Ready**: Optimized for Kaggle GPU environments with dataset `medprs-dataset`.

---

## 📁 Repository Structure

```
.
├── dataset_loader.py          # Data loader for journal metadata & paper splits
├── stage0_parser.py           # Stage 0: PICO, Study Type & Soft Domain Score Extractor
├── stage1_retriever.py        # Stage 1: BioBERT SimCPSR Dense Retriever (FAISS Index)
├── stage2_hybrid_gate.py      # Stage 2: Python Integrity Gate & Soft Domain Matcher
├── stage3_dear_reranker.py    # Stage 3: DeAR Listwise Reranker & Reasoning Trace
├── evaluator.py               # Automated evaluation metrics (Recall, Acc@K, NDCG, Kendall Tau)
├── run_kaggle_pipeline.py     # Main end-to-end execution script
├── Pipeline_Paper_To_Journal_Y_Te_v2.md  # Master v2.1 Pipeline Specification
├── DTAR_Pipeline_MedPRS.md    # DTAR Architecture Specification
├── journal_full_info.csv      # 1,406 PubMed journal metadata dataset (22 columns)
└── README.md
```

---

## 🛠️ Quick Start

### 1. Installation & Environment Setup
```bash
pip install torch transformers faiss-cpu pandas numpy scipy scikit-learn
```

### 2. Run End-to-End Pipeline Demo
```bash
python run_kaggle_pipeline.py
```

### 3. Run Benchmark on Kaggle
Set `data_dir="/kaggle/input/medprs-dataset/"` in `run_kaggle_pipeline.py` and run on Kaggle GPU Notebook.
