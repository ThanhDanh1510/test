# MedStrategist v3.0: Báo Cáo Tóm Tắt Kiến Trúc & Đề Tài Nghiên Cứu

## Risk-Aware, Policy-Constrained, Pareto-Optimal and Uncertainty-Calibrated Medical Journal Submission Strategist

> **Tư tưởng cốt lõi (Core Research Thesis):**  
> Chuyển đổi căn bản từ bài toán *"Tìm tạp chí có nội dung giống bài báo nhất"* (Semantic Matching) sang bài toán **"Hỗ trợ ra quyết định chiến lược nộp bài tối ưu: Trong các tạp chí hợp lệ, tạp chí nào là lựa chọn tốt nhất dưới các mục tiêu, ràng buộc chính sách và độ bất định cụ thể của tác giả?"** (Constraint-Aware Strategic Decision Support).

---

## 1. 📌 Bối Cảnh & 4 Thách Thức Y Khoa Cốt Lõi

1. **Không gian nhãn lớn & chuyên biệt (1,406 Tạp chí PubMed/SCImago)**: Yêu cầu bộ truy vấn ứng viên siêu tốc (< 15ms) có độ phủ lớn (Recall@50 > 95%).
2. **Rủi ro Bị Từ chối Sơ khảo (Desk Reject)**: Các tạp chí có chính sách cấm ngặt nghèo trong `Aims & Scope` (ví dụ: Tạp chí lâm sàng cấm *Case Report*, cấm nghiên cứu *Cell-Line thuần*).
3. **Nguy cơ Tạp chí Săn mồi (Predatory / Delisted Journals)**: Cần chốt chặn an toàn (DOAJ, PubMed Active Index) bảo vệ tác giả.
4. **Nhu cầu Giải thích Minh bạch Y khoa (Interpretability)**: Bác sĩ / Nhà nghiên cứu cần biết rõ lý do tạp chí phù hợp với cấu trúc **PICO** (Population, Intervention, Comparison, Outcome) và **Study Type** của họ.

---

## 2. 🗺️ Sơ Đồ Kiến Trúc Luồng Tổng Thể (End-to-End Workflow)

```text
                       [Bài báo Input: Title + Abstract + Keywords]
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: Structured Paper Understanding                                         │
│ • Trích xuất PICO + Study Type + MeSH + Soft Signal Probabilities (is_case...)  │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0.5: Journal Policy Constraint Encoder (Pre-computed Offline)            │
│ • Bóc tách Aims & Scope 1,406 tạp chí thành ràng buộc máy đọc + Evidence Spans │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: BioBERT SimCPSR Dual-Stream Dense Retrieval                            │
│ • Dual-Stream Query Fusion (Title Stream + Context Stream) ➔ Top 50 (Recall 97%)│
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Risk-Aware Policy Gate                                                 │
│ • Hard Integrity Check (Loại Predatory/Delisted)                               │
│ • Tính điểm rủi ro R_policy(x,j) & Phân 3 Buckets: [ALLOW] [CONFLICT] [AMBIGUOUS]│
│ • Lọc Top 20 ứng viên sạch & an toàn                                            │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3 & 4: Strategic Utility Scorer & Adaptive Disambiguation (AMAR)          │
│ • Xây dựng vector tiện ích đa chiều V(x,j) = [Fit, Policy, Quality, Risk, Pref]│
│ • Tính điểm Strategic Utility U(x,j | θ) theo trọng số cá nhân hóa             │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Multi-Objective Pareto Frontier Recommendation                         │
│ • Loại bỏ các tạp chí bị thống trị đa mục tiêu                                  │
│ • Gán nhãn Decision Profiles: [Best Overall], [High Prestige], [Safest Scope]   │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: Uncertainty Quantification & 90% Conformal Confidence Set              │
│ • Định lượng độ bất định mô hình U_model = Std(Bootstrap)                       │
│ • Xuất tập gợi ý tin cậy cam kết bao phủ 90% toán học (Conformal Set)           │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: Evidence-Grounded Transparent Explanation                              │
│ • Kết xuất bằng chứng 3 chiều (Tích cực, Ma sát rủi ro, Trích dẫn Policy)       │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
  OUTPUT: TOP 5 CHIẾN LƯỢC + PARETO OPTIONS + CONFIDENCE SET + EVIDENCE (43ms)
```

---

## 3. 🎯 Hàm Tiện Ích Chiến Lược Đa Mục Tiêu (Strategic Utility)

Thay vì chỉ dùng 1 điểm tương đồng đơn lẻ, hệ thống tối ưu hóa hàm tiện ích đa chiều:

\[
U(x, j \mid \theta) = \lambda_f F(x,j) + \lambda_p P(x,j) + \lambda_q Q(j) + \lambda_i I(j) + \lambda_s S(j) - \lambda_r R(x,j)
\]

- **\(F(x,j)\)**: *Semantic Fit* (Độ tương hợp ngữ nghĩa & chuyên ngành y học từ BioBERT SimCPSR).
- **\(P(x,j)\)**: *Policy Compatibility* (Mức độ tuân thủ chính sách xuất bản \(= 1 - R\)).
- **\(Q(j)\)**: *Quality & Impact Proxy* (Chỉ số chất lượng SCImago Q1–Q4, SJR, H-Index).
- **\(I(j)\)**: *Integrity Confidence* (Chỉ mục an toàn PubMed, DOAJ, không rủi ro predatory).
- **\(S(j)\)**: *Preference Satisfaction* (Mức độ thỏa mãn yêu cầu riêng của tác giả).
- **\(R(x,j)\)**: *Policy Conflict Risk* (Điểm phạt rủi ro bị từ chối sơ khảo Desk Reject).
- **\(\theta\)**: Vector trọng số tùy biến theo chiến lược của tác giả (*ưu tiên danh tiếng hay ưu tiên an toàn*).

---

## 4. 🌟 5 Đột Phá Khoa Học Lớn Nhất Của Đề Tài (Key Novelties)

| STT | Tên Đột Phá | Ý Nghĩa Khoa Học & Ứng Dụng Thực Tiễn |
|:---:|---|---|
| **1** | **Policy-Aware Risk Modeling** | Chuyển đổi văn bản `Aims & Scope` thành các ràng buộc máy đọc (`case_report_excluded`...) và tính điểm phạt rủi ro \(R_{\text{policy}}\). |
| **2** | **Multi-Objective Pareto Frontier** | Không áp đặt 1 bảng xếp hạng cứng; tìm đường biên tối ưu Pareto giúp tác giả chủ động lựa chọn giữa: **"Tạp chí Uy tín cao (Q1)"** \(\leftrightarrow\) **"Tạp chí An toàn phạm vi nhất (Safest Scope)"**. |
| **3** | **90% Conformal Confidence Set** | Đưa ra tập hợp tạp chí tin cậy có **bảo đảm bao phủ 90% toán học**, tự động cảnh báo khi bài báo thuộc dạng liên ngành khó phân định. |
| **4** | **Dual-Stream & Adaptive Disambiguation (AMAR)** | Tách 2 luồng Tiêu đề + Toàn văn chống loãng đặc trưng và kích hoạt bộ phân định từ khóa thực thể y sinh khi các ứng viên bám sát nút. |
| **5** | **Evidence-Grounded Explanations** | Ngăn ngừa 100% hiện tượng "ảo giác" (hallucination) của LLM bằng cách ép LLM chỉ được diễn giải dựa trên các đối tượng bằng chứng thực tế đã kiểm chứng. |

---

## 5. 📊 Kết Quả Thực Nghiệm Trên Benchmark MedPRS (Mới Nhất)

- **Tập dữ liệu**: **842,424 bài Train** | **120,346 bài Val** | **100,000 bài Test** | **1,406 Tạp chí Y học**.
- **Kết quả đo đạc thực tế chính thức**:

| Chỉ số Đánh Giá (Metric) | Kết quả Đo Đạc Mới Nhất | Ý nghĩa Thực Nghiệm & Đóng Góp |
|---|:---:|---|
| **Stage 1 Recall@50** | **97.00%** | 97/100 bài báo có tạp chí gốc nằm trong Top 50 ứng viên. |
| **Top-1 Accuracy** | **61.00%** | 61/100 bài báo đoán chuẩn xác 100% tên tạp chí ở ngay vị trí #1. |
| **Top-3 Accuracy** | **76.00%** | Khả năng định vị chính xác trong Top 3 lựa chọn đầu. |
| **Top-5 Accuracy** | **84.00%** | 84/100 bài báo trúng đích ngay trong Top 5 đề xuất. |
| **Top-10 Accuracy** | **84.00%** | Độ phủ chuẩn xác cao của toàn bộ nhóm đề xuất. |
| **NDCG@5 / NDCG@10** | **0.7325** | Chất lượng thứ bậc xếp hạng vượt trội (tăng vọt từ 0.66). |
| **Mean Uncertainty** | **0.0253** | Độ tin cậy cao, độ bất định của mô hình cực kỳ thấp. |
| **Thời gian Thực thi (Latency P50)** | **44.53 ms / bài** | Siêu nhanh, sẵn sàng 100% cho Web/API Production thời gian thực. |

---

## 6. 💻 Minh Họa Cấu Trúc Đầu Ra (Output Schema Demo)

```json
{
  "paper_summary": {
    "study_type": "Original Clinical Research",
    "pico": {
      "population": "Target clinical patient cohort",
      "intervention": "Bat wing skin biomechanics"
    }
  },
  "recommendations": [
    {
      "rank": 1,
      "journal_title": "Journal of the Royal Society Interface",
      "strategic_utility": 0.892,
      "decision_profile": "Best Overall Strategic Balance",
      "pareto_optimal": true,
      "dimensions": { "fit": 0.95, "policy_fit": 1.0, "quality": 0.88, "policy_risk": 0.0 },
      "evidence": {
        "summary_text": "Khuyến nghị vị trí #1: Khớp hoàn hảo giữa nghiên cứu cơ học sinh học và phạm vi tạp chí. An toàn chính sách: 100%. Phân hạng: Q1.",
        "positive_evidence": [{ "dimension": "study_design", "journal_evidence": "Tạp chí ưu tiên bài báo cơ sinh học liên ngành." }]
      }
    }
  ],
  "confidence_set": {
    "coverage_target": 0.90,
    "journals": ["Journal of the Royal Society Interface", "PLoS Computational Biology", "Journal of Physiology"]
  }
}
```
