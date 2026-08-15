# DTAR-Submission Strategist v3.0

## Risk-Aware, Counterfactual, Pareto and Uncertainty-Calibrated Medical Journal Recommendation

**DTAR v3.0** transforms medical journal recommendation from simple semantic matching into a **risk-aware, policy-constrained, and uncertainty-calibrated strategic decision support system** for biomedical researchers matching papers against 1,406 PubMed/SCImago indexed journals.

---

## 🚀 Key Architectural Innovations (v3.0)

1. **Strategic Utility Optimization $U(x, j \mid \theta)$**: Jointly balances Semantic Fit ($F$), Policy Compatibility ($P$), Journal Quality/Impact ($Q$), Indexing Integrity ($I$), User Preferences ($U$), and Policy Conflict Risk ($R$).
2. **Stage 0.5 Journal Policy Constraint Extraction**: Converts unstructured `Aims & Scope` into machine-verifiable constraints (`article_types`, `excluded_types`, `evidence_spans`).
3. **Risk-Aware Policy Gate**: Categorizes candidates into `ALLOW`, `CONFLICT`, and `AMBIGUOUS` with calibrated policy conflict risk $R_{\text{policy}}(x,j)$.
4. **Pareto Frontier Recommendation**: Filters out dominated candidate venues and assigns actionable Decision Profiles (*"Best Overall Balance"*, *"High Prestige"*, *"Safest Scope"*).
5. **Calibrated Uncertainty & 90% Conformal Confidence Set**: Quantifies score uncertainty via bootstrap ensemble perturbations and produces an empirical 90%-coverage recommendation set.
6. **Evidence-Grounded Explanations**: Generates transparent, verifiable positive/negative/policy evidence traces directly grounded in journal metadata.

---

## 📁 Repository Structure

```
.
├── dataset_loader.py               # Data loader for 1,406 PubMed journals & MedPRS paper splits
├── stage0_parser.py                # Stage 0: PICO, Study Type & Soft Signal Extractor
├── stage0_5_policy_encoder.py      # Stage 0.5: Journal Policy Constraint & Evidence Span Encoder
├── stage1_retriever.py             # Stage 1: BioBERT SimCPSR Dense Retriever (Top-50 Recall@50 = 96%)
├── stage2_hybrid_gate.py           # Stage 2: Risk-Aware Policy Gate & Soft Domain Matcher
├── stage3_strategic_scorer.py      # Stage 3 & 4: Strategic Utility Vector & Multi-Objective Scorer
├── stage5_pareto_recommender.py    # Stage 5: Multi-Objective Pareto Frontier Recommender
├── stage6_uncertainty.py           # Stage 6: Uncertainty Layer & 90% Conformal Confidence Set
├── stage7_evidence_explainer.py    # Stage 7: Evidence-Grounded Explanation Generator
├── evaluator.py                    # Multi-Metric Benchmark Evaluator (Recall, NDCG, Uncertainty, Latency)
├── run_kaggle_pipeline.py          # Master End-to-End Orchestrator
├── train_simcprs.py                # Multi-GPU FP16 Continuous Trainer (Epoch 2 -> 10)
├── DTAR_Submission_Strategist_v3.0.md # Master v3.0 System Architecture Specification
├── journal_full_info.csv           # 1,406 PubMed journal metadata dataset (22 columns)
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
