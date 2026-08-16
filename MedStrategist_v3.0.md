# MedStrategist v3.0

## Risk-Aware, Counterfactual, Pareto and Uncertainty-Calibrated Medical Journal Submission Strategist

> **Core idea:** Không còn chỉ trả lời *"Tạp chí nào giống bài này nhất?"*. Hệ thống trả lời *"Trong các tạp chí hợp lệ, tạp chí nào là lựa chọn CHIẾN LƯỢC TỐT NHẤT cho bài này dưới các mục tiêu, ràng buộc chính sách và độ bất định cụ thể?"*

---

# 1. Research Positioning

## 1.1 Bài toán cũ

Pipeline v2.1 chủ yếu giải:

\[
\hat j = \arg\max_j \; Match(x,j)
\]

với một cascade:

```text
Paper
  ↓
PICO / Study Type / MeSH
  ↓
Dense Retrieval
  ↓
Hard/Soft Policy Gate
  ↓
DeAR Reranking
  ↓
Explanation / Faithfulness
  ↓
Top-5 Journals
```

Kiểu formulation này hợp lý nhưng vẫn gần với **association/content-based venue recommendation**: journal tốt là journal có mức tương đồng cao với paper.

## 1.2 Bài toán mới

DTAR v3.0 chuyển thành:

> **Constraint-aware strategic venue recommendation**: chọn một journal có utility cao nhất khi đồng thời xét semantic fit, policy compatibility, journal utility, user preferences và uncertainty.

Ta định nghĩa:

\[
U(x,j\mid\theta)
=
\lambda_f F(x,j)
+\lambda_q Q(j)
+\lambda_s S(j)
+\lambda_i I(j)
+\lambda_p P(x,j)
-\lambda_r R(x,j)
-\lambda_c C(j)
\]

Trong đó:

- \(F\): semantic/domain fit.
- \(Q\): journal quality/impact proxy.
- \(S\): strategic utility theo user preference.
- \(I\): integrity/indexing confidence.
- \(P\): policy compatibility.
- \(R\): policy-conflict / submission-risk score.
- \(C\): cost hoặc các penalty tùy metadata sẵn có.
- \(\theta\): preference vector của người dùng.

**Lưu ý quan trọng:** với dataset hiện tại, hệ thống **không được gọi là acceptance-probability predictor** nếu không có dữ liệu submission/rejection thực tế. `R(x,j)` trong phiên bản này được định nghĩa là **policy-conflict risk / submission-risk proxy**, không phải xác suất acceptance.

---

# 2. Research Gap & Novelty Strategy

## 2.1 Những gì đã có trong literature

### Publication venue recommendation

Poincare đã reformulate venue recommendation thành treatment-effect estimation, coi venue là treatment và citation/impact là outcome. Đây là bằng chứng rằng việc tách **"venue phù hợp"** khỏi **"venue tạo ra outcome mong muốn"** là một hướng nghiên cứu có cơ sở. [Sato et al., Journal of Informetrics, 2022](https://doi.org/10.1016/j.joi.2022.101283).

### Explainable journal recommendation

Các công trình gần đây đã nghiên cứu journal recommendation có explanation trong biomedical domain, sử dụng topic, related articles và relevant terms. Vì vậy **"thêm explanation" riêng lẻ không nên được coi là novelty chính**. [de Campos et al., 2024](https://doi.org/10.1007/s11257-024-09400-6).

### Hybrid venue recommendation

Các phương pháp 2026 đã kết hợp content và collaborative information cho publication venue recommendation. Vì vậy chỉ thay dense retriever bằng hybrid recommender cũng chưa đủ tạo novelty mạnh. [Knowledge and Information Systems, 2026](https://doi.org/10.1007/s10115-026-02749-7).

### Uncertainty-aware recommendation

GUIDER (AAAI 2026) cho thấy uncertainty có thể được dùng trực tiếp để điều chỉnh reranking/recommendation. Do đó uncertainty riêng lẻ cũng không phải novelty độc quyền của pipeline này. [Xu et al., AAAI 2026](https://doi.org/10.1609/aaai.v40i19.38639).

### Position bias trong listwise LLM reranking

Các hướng 2026 đã nghiên cứu trực tiếp permutation-invariant listwise reranking. Vì vậy **Dynamic Permutation Check** của v2.1 nên được giữ như một robustness module, không phải contribution trung tâm. [InvariRank, 2026](https://arxiv.org/abs/2604.27599).

## 2.2 Khoảng trống mà v3.0 nhắm tới

Pipeline mới không tuyên bố một kỹ thuật đơn lẻ là "mới tuyệt đối". Contribution được đặt ở **problem formulation + integrated system objective + evaluation protocol**:

1. **Policy-aware strategic recommendation**: journal không chỉ được đánh giá bằng semantic similarity mà bằng compatibility với policy của chính journal.
2. **Risk-aware ranking thay vì hard filtering thuần túy**: policy conflict được biến thành một chiều rủi ro có thể calibration và trade-off.
3. **Counterfactual-style utility scoring**: đánh giá *nếu paper được cân nhắc ở journal J* thì utility chiến lược kỳ vọng là bao nhiêu, thay vì chỉ học association paper → journal.
4. **Pareto frontier**: không ép mọi user dùng một ranking duy nhất khi mục tiêu (impact, fit, policy safety, quality) xung đột nhau.
5. **Conformal/uncertainty layer**: thay vì chỉ xuất top-5, hệ thống xuất thêm `confidence_set` để biểu diễn mức độ chắc chắn của lựa chọn.
6. **Evidence-grounded recommendation**: mỗi chiều score phải truy ngược được về evidence cụ thể từ journal metadata/policy, tránh biến explanation thành prose hậu nghiệm.

> **Research thesis:** Journal recommendation nên được xem như một **decision-support problem under policy and uncertainty**, không chỉ là semantic retrieval problem.

---

# 3. Dataset Compatibility — Không cần dataset mới để chạy v1

## 3.1 Dữ liệu hiện tại

Nguồn v2.1 có:

- 842,424 train papers.
- 120,346 validation papers.
- 100,000 test papers.
- 1,406 journals theo problem statement.
- Journal metadata gồm tên, Aims & Scope, SCImago quartile, SJR, H-Index và 9 domain.

## 3.2 Tận dụng dữ liệu hiện có

### Supervised venue label

Mỗi paper đã có venue/journal thực tế của nó:

```text
paper_x → observed_journal
```

Đây là nhãn chính cho retrieval/ranking evaluation.

### Journal profile

Tạo một canonical profile cho journal:

```json
{
  "journal_id": "...",
  "title": "...",
  "aims_scope": "...",
  "domains": ["..."],
  "quartile": "Q1",
  "sjr": 2.15,
  "h_index": 120
}
```

### Policy constraints

Từ Aims & Scope hiện tại, dùng LLM/regex để trích xuất:

```json
{
  "article_types": ["original research", "review", "clinical trial"],
  "excluded_types": ["case report"],
  "population_constraints": [...],
  "method_constraints": [...],
  "domain_constraints": [...],
  "geography_constraints": [...],
  "other_constraints": [...],
  "evidence_spans": [...]
}
```

**Không tạo claim rằng đây là rejection label thực tế.** Đây là `policy_constraint` và được dùng để suy ra `policy_conflict_risk`.

---

# 4. Kiến trúc DTAR v3.0

```text
                                INPUT PAPER
                         Title + Abstract + Keywords
                                      │
                                      ▼
                    ┌─────────────────────────────┐
                    │ STAGE 0: Structured Parsing │
                    │ PICO / Study Type / MeSH    │
                    │ Domains / Research signals  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ STAGE 0.5: Policy Knowledge  │
                    │ Journal constraint encoding │
                    │ pre-computed offline        │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ STAGE 1: Candidate Retrieval │
                    │ BioBERT / dense retrieval    │
                    │ Top-50                      │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          Semantic Fit View              Policy Compatibility View
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ STAGE 2: Risk-aware Gate     │
                    │ Hard integrity constraints   │
                    │ Soft policy conflict score   │
                    │ Preference constraints      │
                    └──────────────┬──────────────┘
                                   │
                                Top-15/20
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ STAGE 3: Strategic Scoring   │
                    │ Fit + Policy + Quality       │
                    │ + Utility + Risk             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ STAGE 4: Counterfactual      │
                    │ Utility Estimator            │
                    │ paper × candidate journal    │
                    └──────────────┬──────────────┘
                                   │
                              utility vector
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ STAGE 5: Pareto Frontier     │
                    │ remove dominated journals    │
                    │ personalize by preferences  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ STAGE 6: Uncertainty Layer   │
                    │ Ensemble / bootstrap /       │
                    │ conformal confidence set     │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ STAGE 7: Evidence-grounded   │
                    │ explanation + decision trace │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
        TOP-5 + PARETO OPTIONS + CONFIDENCE SET + EVIDENCE + RISKS
```

---

# 5. Stage 0 — Structured Paper Understanding

Giữ Single-Pass Parsing của v2.1 nhưng đổi output thành feature schema phục vụ decision model.

```json
{
  "pico": {
    "population": "...",
    "intervention": "...",
    "comparison": "...",
    "outcome": "..."
  },
  "study_type": "Randomized Controlled Trial",
  "mesh_terms": ["..."],
  "domains": {
    "endocrinology": 0.92,
    "cardiology": 0.31
  },
  "paper_signals": {
    "is_case_report": 0.02,
    "is_cell_line": 0.03,
    "is_review": 0.01,
    "is_clinical_trial": 0.94
  }
}
```

### Nguyên tắc

- Không binary hóa tín hiệu khi chưa cần.
- Lưu probability/soft score để downstream model có thông tin uncertainty.
- Cache toàn bộ parsed representation.

---

# 6. Stage 0.5 — Journal Policy Encoder

Đây là module mới quan trọng.

## 6.1 Offline

Chạy một lần trên 1,406 journal:

```text
Aims & Scope
   ↓
LLM extraction
   ↓
Structured policy JSON
   ↓
Embedding per constraint
   ↓
Cache
```

## 6.2 Constraint taxonomy

```text
ARTICLE_TYPE
POPULATION
INTERVENTION / METHOD
DISEASE / DOMAIN
DATA_TYPE
STUDY_DESIGN
GEOGRAPHIC_SCOPE
EXPLICIT_EXCLUSION
OTHER_REQUIREMENT
```

## 6.3 Evidence span

Mỗi constraint phải lưu:

```json
{
  "constraint": "case_reports_excluded",
  "value": true,
  "source_text": "...",
  "source_field": "aims_scope"
}
```

Điều này tạo nền cho **evidence-grounded explanation**.

---

# 7. Stage 1 — Dense Candidate Retrieval

Giữ BioBERT SimCPSR + FAISS làm baseline và candidate generator.

### Mục tiêu

```text
1,406 journals → Top-50
```

### Metrics

- Recall@10
- Recall@20
- Recall@50
- MRR

### Không dùng Stage 1 để quyết định final rank.

Stage 1 chỉ cần đảm bảo journal thật nằm trong candidate set.

---

# 8. Stage 2 — Risk-aware Policy Gate

## 8.1 Hard integrity gate

Loại ngay các journal không đạt yêu cầu integrity/indexing.

```text
Integrity invalid → DROP
```

## 8.2 Policy conflict score

Không loại toàn bộ journal chỉ vì một soft mismatch.

Định nghĩa:

\[
R_{policy}(x,j)
=\sum_k w_k c_k(x,j)
\]

trong đó:

- \(c_k\in[0,1]\): mức xung đột với constraint k.
- \(w_k\): trọng số theo độ nghiêm trọng của constraint.

Ví dụ:

```text
case report + journal explicitly excludes case reports
→ high conflict

clinical trial + journal accepts clinical trials
→ low conflict

methodology partially outside scope
→ medium conflict
```

## 8.3 Ambiguity bucket

Không ép mọi constraint thành 0/1.

```text
ALLOW
CONFLICT
AMBIGUOUS
```

`AMBIGUOUS` được truyền xuống Stage 3 thay vì loại thẳng.

---

# 9. Stage 3 — Strategic Scoring Model

Thay vì chỉ một score similarity, xây vector:

\[
V(x,j) = [F, P, Q, I, R, U]
\]

Trong đó:

- `F`: semantic fit.
- `P`: policy compatibility.
- `Q`: journal quality proxy.
- `I`: integrity.
- `R`: policy conflict risk.
- `U`: user preference satisfaction.

## 9.1 Baseline scoring

```python
score = (
    0.40 * fit
  + 0.20 * policy_fit
  + 0.15 * quality
  + 0.10 * preference_fit
  + 0.05 * integrity
  - 0.10 * policy_risk
)
```

**Các hệ số trên chỉ là baseline khởi đầu, không phải hyperparameter cuối cùng.** Hệ số chính thức phải được học/tối ưu trên validation set.

## 9.2 Learning-to-rank

Training examples:

```text
paper x
positive = observed journal
hard negatives = Top-K journals retrieved by Stage 1
```

Dùng pairwise/listwise ranking loss.

### Hard negative design

Ưu tiên các negative journal:

1. semantic similarity cao;
2. cùng domain nhưng policy conflict;
3. cùng quartile nhưng policy mismatch;
4. gần điểm ranking.

Điểm này quan trọng vì model sẽ học được sự khác nhau giữa:

> **"rất giống"** và **"thực sự nên gửi"**.

---

# 10. Stage 4 — Counterfactual-Style Utility Estimator

## 10.1 Mục tiêu

Không claim causal acceptance nếu không có randomized/submission data.

Thay vào đó học một **conditional utility model**:

\[
\hat U(x,j)=f_\phi(x,j,m_j)
\]

với:

- \(x\): paper representation.
- \(j\): journal.
- \(m_j\): journal metadata.

## 10.2 Outcome proxy hiện tại

Có thể xây utility target từ metadata có sẵn:

\[
Y_{proxy}(j)
=
\alpha_1 Norm(SJR)
+
\alpha_2 Norm(HIndex)
+
\alpha_3 QuartileScore
\]

sau đó điều chỉnh bằng paper-journal fit.

Tuy nhiên cần ghi rõ đây là:

> **strategic utility proxy, not causal publication outcome**.

## 10.3 Causal extension cho phiên bản future

Nếu sau này thu thập được:

- submission logs,
- desk-reject labels,
- acceptance labels,
- time-to-decision,
- citation outcomes,

thì có thể chuyển sang:

\[
Y_j(x) = E[Y \mid do(J=j), X=x]
\]

và áp dụng doubly robust / propensity-based treatment effect estimation.

Đây là đường nâng cấp khoa học tự nhiên của hệ thống, nhưng **không đưa causal claim vào v1 nếu dữ liệu chưa đủ**.

---

# 11. Stage 5 — Pareto Recommendation

Thay vì bắt mọi user nhận cùng một Top-5, xem mỗi candidate là vector:

\[
J=(Fit, Policy, Quality, Risk, Preference)
\]

Journal A dominates B nếu A không tệ hơn B trên mọi chiều và tốt hơn ít nhất một chiều.

## 11.1 Pareto frontier

```text
Quality ↑
       │
  J1   │        J2
       │
       │   J3
       │
 J5    │
       └────────────────→ Policy-safe / Fit
```

Chỉ các journal không bị dominance được giữ làm **strategic options**.

## 11.2 Personalized utility

User có preference vector:

```json
{
  "min_quartile": "Q1",
  "impact_weight": 0.35,
  "fit_weight": 0.25,
  "risk_weight": 0.25,
  "speed_weight": 0.15
}
```

Từ frontier, tính:

\[
J^* = \arg\max_j U(j\mid\theta)
\]

Kết quả không còn là một ranking cứng cho tất cả users.

---

# 12. Stage 6 — Uncertainty & Risk Calibration

## 12.1 Ba loại uncertainty

### Model uncertainty

Nhiều scoring heads / bootstrap models disagree.

### Data ambiguity

Paper interdisciplinary, policy không rõ, profile sparse.

### Ranking uncertainty

Top candidates có utility rất sát nhau.

---

# 13. Recommendation Confidence

Cho mỗi journal:

```json
{
  "utility": 0.81,
  "uncertainty": 0.07,
  "policy_risk": 0.12,
  "ranking_margin": 0.03
}
```

Định nghĩa uncertainty đơn giản ở bản đầu:

\[
U_{model}(x,j)=Std(\hat y_1, ..., \hat y_B)
\]

với B bootstrap models.

Không cần M ensemble khổng lồ; 5 models là đủ cho baseline thực nghiệm.

---

# 14. Conformal Confidence Set

Đây là extension có tiềm năng làm contribution kỹ thuật riêng.

Thay vì chỉ output:

```text
Top-5
```

thêm:

```text
90%-confidence recommendation set
```

## 14.1 Calibration

Trên validation:

1. Fit base ranker trên train.
2. Tính nonconformity score cho journal thật.
3. Chọn quantile \(q_{0.9}\).
4. Với paper mới, giữ journal có nonconformity ≤ threshold.

Mục tiêu:

\[
P(J_{true}\in C(X)) \ge 1-\alpha
\]

trong điều kiện calibration assumptions phù hợp.

## 14.2 Ý nghĩa UI

```text
Top-1: Journal A
Top-5: A, B, C, D, E
Confidence Set (90%): A, B, C, D, E, F, G
```

Nếu set rất lớn → hệ thống cảnh báo paper khó phân biệt.

Nếu set nhỏ → recommendation đáng tin cậy hơn.

---

# 15. Stage 7 — Evidence-grounded Explanation

Không dùng reasoning trace tự do làm nguồn sự thật.

Mỗi explanation phải được render từ các evidence objects:

```json
{
  "journal": "Journal A",
  "positive_evidence": [
    {
      "dimension": "study_type",
      "paper_signal": "Randomized Controlled Trial",
      "journal_evidence": "Accepts clinical trials"
    }
  ],
  "negative_evidence": [
    {
      "dimension": "population",
      "risk": 0.22,
      "reason": "population only partially represented in scope"
    }
  ],
  "policy_evidence": [
    {
      "constraint": "case reports",
      "value": "excluded"
    }
  ]
}
```

Sau đó LLM chỉ có nhiệm vụ **verbalize evidence đã cấu trúc**.

### Nguyên tắc

```text
Retriever / policy parser → evidence
                              ↓
                      deterministic score
                              ↓
                       explanation LLM
```

Không làm ngược lại:

```text
LLM explanation
      ↓
cosine similarity
      ↓
claim faithfulness
```

---

# 16. Final Output Schema v3.0

```json
{
  "paper_summary": {
    "study_type": "Randomized Controlled Trial",
    "pico": {
      "population": "T2D patients",
      "intervention": "SGLT2 inhibitors",
      "comparison": "Placebo",
      "outcome": "CV mortality reduction"
    }
  },
  "recommendations": [
    {
      "rank": 1,
      "journal_title": "Journal A",
      "strategic_utility": 0.86,
      "dimensions": {
        "fit": 0.92,
        "policy_fit": 0.95,
        "quality": 0.89,
        "preference_fit": 0.81,
        "policy_risk": 0.08
      },
      "uncertainty": 0.05,
      "pareto_optimal": true,
      "decision_profile": "Best overall balance",
      "evidence": {
        "positive": [...],
        "negative": [...]
      },
      "confidence_flags": {
        "low_confidence": false,
        "needs_review": false
      }
    }
  ],
  "confidence_set": {
    "coverage_target": 0.90,
    "journals": ["A", "B", "C", "D", "E"]
  },
  "system_notes": {
    "policy_data_version": "...",
    "model_version": "..."
  }
}
```

---

# 17. Evaluation Protocol — Quan trọng hơn Architecture

## 17.1 Retrieval

| Metric | Mục tiêu |
|---|---:|
| Recall@10 | ≥ 0.90 |
| Recall@50 | ≥ 0.95 |
| MRR | report |

**Không tự đặt mục tiêu bắt buộc trước thực nghiệm nếu baseline chưa được đo.** Mục tiêu trên là target engineering, không phải kết quả đã đạt.

## 17.2 Ranking

| Metric | Ý nghĩa |
|---|---|
| NDCG@5 | ranking quality |
| MRR | first relevant journal |
| Hit@1 / Hit@5 | top recommendation |
| Recall@5 | coverage |

## 17.3 Risk

Đánh giá policy-risk trên benchmark annotation thủ công nhỏ:

```text
500–1,000 paper × journal pairs
```

Chia:

- ALLOW
- CONFLICT
- AMBIGUOUS

Metrics:

- Macro-F1
- Precision conflict
- Recall conflict
- False-safe-rate

### Quan trọng

Không đánh tráo:

```text
policy conflict accuracy
```

với:

```text
real desk-reject prediction accuracy
```

Hai bài toán khác nhau.

## 17.4 Calibration

- ECE
- Brier score
- Reliability diagram
- selective risk / coverage curve

## 17.5 Pareto usefulness

Đánh giá:

- tỷ lệ recommendation bị dominated;
- utility regret so với oracle;
- diversity của Pareto options;
- user-weight sensitivity.

## 17.6 Confidence set

Nếu dùng conformal layer:

- empirical coverage;
- average set size;
- coverage-vs-size curve.

Mục tiêu ví dụ:

```text
Target coverage = 90%
Empirical coverage ≈ 90%
```

Sai số nhỏ là chấp nhận được; không claim exact guarantee ngoài assumptions.

---

# 18. Ablation Study — Bắt buộc để chứng minh contribution

## A0 — Existing baseline

```text
BioBERT → FAISS → Hard/Soft Gate → DeAR
```

## A1 — + Policy encoder

Đo policy conflict.

## A2 — + Risk-aware scoring

Đo NDCG/utility thay đổi.

## A3 — + Strategic utility

So sánh semantic-only vs strategic ranking.

## A4 — + Pareto frontier

Đo dominance/regret.

## A5 — + Uncertainty

Đo calibration/selective prediction.

## A6 — + Conformal confidence set

Đo coverage và set size.

## A7 — Full v3.0

```text
Retrieval
+ Policy
+ Risk
+ Utility
+ Pareto
+ Uncertainty
+ Evidence
```

---

# 19. Hard Negative Research Design

Đây là phần nên đầu tư để model học được scientific signal.

## Negative Type 1 — Semantic hard negative

Cùng domain, similarity cao.

## Negative Type 2 — Policy hard negative

Semantic fit cao nhưng explicit exclusion.

## Negative Type 3 — Quality hard negative

Q1 journal nhưng scope lệch.

## Negative Type 4 — Preference hard negative

Scope phù hợp nhưng vi phạm user constraints.

## Negative Type 5 — Near-tie hard negative

Hai journal gần nhau theo baseline.

### Hypothesis

Nếu model v3 chỉ giúp Top-1 bằng cách tăng semantic score thì chưa chứng minh contribution.

Model phải đặc biệt cải thiện ở:

> **policy-conflicting hard negatives và near-tie cases**.

---

# 20. Data Construction Strategy

## 20.1 Không cần thêm data toàn cục ngay

Dùng:

```text
Existing paper-journal labels
Existing journal metadata
Existing Aims & Scope
```

## 20.2 Chỉ cần annotation mới quy mô nhỏ

Tạo một benchmark độc lập:

```text
500–1,000 paper-journal pairs
```

với 3 labels:

```text
ALLOW
CONFLICT
AMBIGUOUS
```

và 1 evidence span.

### Vì sao đây là data-efficient?

Bạn không cần hàng chục nghìn expert labels.

Bạn chỉ cần benchmark để đo **policy understanding**, còn ranking supervision tận dụng hàng triệu paper-journal observations hiện có.

---

# 21. Production Architecture

```text
                    ┌───────────────┐
                    │   Paper API   │
                    └───────┬───────┘
                            │
                            ▼
                  Stage 0 Parser Cache
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
        FAISS Journal Search      Policy Store
               │                         │
               └────────────┬────────────┘
                            ▼
                     Candidate Store
                            │
                            ▼
                    Risk-aware Ranker
                            │
                            ▼
                  Utility / Pareto Layer
                            │
                            ▼
                    Uncertainty Layer
                            │
                            ▼
                 Evidence Explanation
                            │
                            ▼
                       API Response
```

### Storage

Journal profile cache:

```text
journal_id
embedding
policy_json
constraint_embeddings
quartile
sjr
h_index
integrity_status
```

### Latency design

Các phần journal-level nên precompute offline:

- journal embedding;
- policy embedding;
- constraints;
- quality normalization;
- integrity flags.

Online chỉ xử lý:

```text
paper parse
→ vector search
→ candidate policy matching
→ rank
→ utility
→ uncertainty
```

Mục tiêu latency nên được **benchmark lại từ implementation thực tế**, không hard-code các con số 2.5s–3.2s của v2.1.

---

# 22. What to Remove from v2.1

## Remove as primary contribution

### Dynamic Permutation Check

Giữ như robustness test, không viết như innovation chính.

### Cosine faithfulness threshold

Không dùng:

```text
LLM scope_fit
→ embedding cosine
→ faithful / hallucinated
```

làm định nghĩa duy nhất của faithfulness.

### Synthetic SetFit 98.5%

Không claim mạnh nếu train/test đều synthetic.

Thay bằng policy-conflict benchmark có held-out real annotation.

### "100% robust / production-grade"

Thay bằng metric và confidence interval.

---

# 23. Main Research Contributions — Dùng để viết paper

## Contribution 1 — New problem formulation

> We formulate medical journal recommendation as a **risk-aware strategic decision problem** rather than pure semantic matching.

## Contribution 2 — Policy-aware recommendation

> We introduce a structured journal-policy representation that converts unstructured journal scope statements into machine-verifiable constraints and evidence.

## Contribution 3 — Strategic utility + Pareto recommendation

> We jointly optimize paper-journal fit, policy compatibility, journal utility and user preferences, and expose the Pareto-optimal venue set rather than a single universal ranking.

## Contribution 4 — Uncertainty-controlled recommendations

> We augment the ranking with calibrated uncertainty and a confidence recommendation set, allowing the system to distinguish decisive cases from intrinsically ambiguous venue choices.

## Contribution 5 — Evidence-grounded decision trace

> We generate explanations from structured evidence objects instead of treating free-form LLM reasoning as ground truth.

---

# 24. Core Research Hypotheses

## H1 — Policy-aware ranking

Adding structured policy compatibility improves ranking on policy-conflicting hard negatives compared with semantic-only ranking.

## H2 — Risk-aware utility

A risk-aware utility objective produces better strategic choices than maximizing semantic similarity alone, especially among near-tie candidates.

## H3 — Pareto recommendation

Pareto filtering reduces dominated recommendations while preserving diverse trade-offs across fit, quality and policy safety.

## H4 — Uncertainty calibration

Uncertainty-aware output is better calibrated than raw ranking scores and can identify ambiguous papers/candidate sets.

## H5 — Evidence grounding

Structured evidence explanations are more faithful to journal policy than unconstrained LLM-generated explanations.

---

# 25. Strong Experimental Story

Paper nên kể câu chuyện theo thứ tự:

```text
Problem:
semantic fit is not enough
        ↓
Observation:
policy and user objectives can conflict with semantic similarity
        ↓
Method:
risk-aware strategic ranking
        ↓
Decision:
Pareto frontier instead of one-size-fits-all Top-5
        ↓
Reliability:
uncertainty + confidence set
        ↓
Trust:
evidence-grounded explanation
```

Đây là story mạnh hơn việc kể:

```text
BioBERT
→ FAISS
→ SetFit
→ DeAR
→ Kendall
→ BGE
```

vì reviewer nhìn thấy **một thesis thống nhất**, không phải danh sách module.

---

# 26. Minimal Implementable Version (MVP Research)

Nếu nguồn lực hạn chế, triển khai đúng 6 module:

```text
1. Existing BioBERT + FAISS
2. Structured policy extraction
3. Policy conflict scorer
4. Risk-aware learning-to-rank
5. Pareto frontier
6. Bootstrap uncertainty
```

Chưa cần causal inference thật.

Chưa cần Conformal ngay.

Chưa cần LLM listwise reranking phức tạp.

Đây là phiên bản có thể chạy với data hiện tại và đủ để kiểm nghiệm thesis.

---

# 27. Recommended v3.0 Milestones

## M1 — Baseline reproduction

- Reproduce v2.1.
- Fix journal count inconsistency: problem statement currently says 1,406 while one pipeline block says 1,408.
- Establish exact Recall@50 and NDCG@5.

## M2 — Policy representation

- Parse Aims & Scope.
- Build constraint schema.
- Create 500–1,000 pair benchmark.

## M3 — Risk-aware ranker

- Generate hard negatives.
- Train pairwise/listwise model.
- Compare with semantic-only baseline.

## M4 — Pareto layer

- Build 4–5 dimensions.
- Evaluate dominated-rate and utility regret.

## M5 — Uncertainty

- Train 5 bootstrap models.
- Compute mean/std.
- Plot reliability diagrams.

## M6 — Optional conformal extension

- Calibrate confidence set.
- Report empirical coverage vs average set size.

## M7 — Full ablation

- A0 → A7.
- Statistical significance test.
- Error analysis on policy hard negatives.

---

# 28. Potential Paper Titles

### Conservative

**Risk-Aware and Policy-Constrained Medical Journal Recommendation with Uncertainty Calibration**

### Stronger

**From Journal Matching to Submission Strategy: Risk-Aware Medical Journal Recommendation with Pareto Decision Support**

### Most distinctive

**Where Should I Submit? Counterfactual-Style, Policy-Aware and Uncertainty-Calibrated Medical Journal Recommendation**

---

# 29. Limitations — Must State Explicitly

1. Không có rejection/submission logs nên không được claim acceptance probability thực tế.
2. Counterfactual utility v1 chỉ là observational/utility proxy, không phải causal effect estimator chuẩn.
3. Journal policy extraction từ text có thể lỗi; cần evidence spans và ambiguity state.
4. Quality metadata như SJR/H-index là proxy cho journal utility, không phải chất lượng paper.
5. Conformal coverage phụ thuộc calibration/exchangeability assumptions.
6. Recommendation cuối cùng vẫn là decision support; tác giả cần kiểm tra submission guidelines hiện hành của journal trước khi nộp.

---

# 30. Final Target Architecture

```text
                         PAPER
                           │
                           ▼
                 Structured Understanding
                           │
                           ▼
                  Dense Candidate Recall
                           │
                           ▼
              ┌────────────────────────────┐
              │ Policy-aware Risk Modeling │
              └─────────────┬──────────────┘
                            │
                            ▼
                Strategic Utility Ranking
                            │
                            ▼
                    Pareto Frontier
                            │
                            ▼
                 Uncertainty Calibration
                            │
                            ▼
               Confidence Recommendation Set
                            │
                            ▼
                Evidence-grounded Explanation
                            │
                            ▼
             TOP-5 + TRADE-OFFS + CONFIDENCE
```

## One-sentence research identity

> **DTAR v3.0 is a decision-support system that transforms medical journal recommendation from semantic matching into a policy-aware, risk-controlled and uncertainty-calibrated strategic submission problem.**

---

# 31. References

1. Sato, R., Yamada, M., Kashima, H. *Poincare: Recommending Publication Venues via Treatment Effect Estimation*. Journal of Informetrics, 16(2), 101283, 2022. https://doi.org/10.1016/j.joi.2022.101283
2. de Campos, L. M., Fernández-Luna, J. M., Huete, J. F. *An explainable content-based approach for recommender systems: a case study in journal recommendation for paper submission*. User Modeling and User-Adapted Interaction, 2024. https://doi.org/10.1007/s11257-024-09400-6
3. *Combining content information with collaborative filtering for publication venue recommendation*. Knowledge and Information Systems, 2026. https://doi.org/10.1007/s10115-026-02749-7
4. Xu, C., Wang, X., Guan, Z., Zhao, W., Yan, M. *GUIDER: Uncertainty Guided Dynamic Re-ranking for Large Language Models Based Recommender Systems*. AAAI 2026. https://doi.org/10.1609/aaai.v40i19.38639
5. Bito, E., Ren, Y., He, E. *One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation*. arXiv, 2026. https://arxiv.org/abs/2604.27599
6. *A Survey on Causal Inference for Recommendation*. The Innovation, 2024. https://doi.org/10.1016/j.xinn.2024.100590

---

# 32. Source Mapping from v2.1

This v3.0 intentionally preserves the strongest ideas from the original architecture:

- Single-pass PICO/Study-Type parsing.
- BioBERT + FAISS candidate retrieval.
- Integrity hard gate.
- Soft domain/policy scoring.
- DeAR as an optional expensive reranker/robustness baseline.
- Confidence flags.

The main change is **not adding more modules for their own sake**. It is changing the optimization target from:

\[
\text{semantic match}
\]

to:

\[
\text{strategic utility under policy constraints and uncertainty}
\]

That is the central research direction of v3.0.
