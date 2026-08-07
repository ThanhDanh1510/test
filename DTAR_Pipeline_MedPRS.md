# DTAR-Slim: Pipeline Kết hợp MedPRS – Journal Integrity Gate – DeAR

> **Phiên bản Hợp nhất Tối ưu**: Kế thừa toàn bộ thế mạnh về khả năng giải thích (Reasoning Trace) và kiểm tra an toàn xuất bản (Journal Integrity Gate) của **DTAR**, đồng thời loại bỏ triệt để sự chồng chéo cồng kềnh của TourRank để đạt thời gian phản hồi **< 2.5 giây**.

---

## 1. Sơ đồ Kiến trúc Hợp nhất DTAR-Slim

```
                       [Bài báo Input: Title + Abstract + Keywords]
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: Single-Pass Fast LLM Parsing & Normalization                           │
│ • 1 Call LLM duy nhất: Trích xuất PICO + Study Type + MeSH + 9 Domain Flags      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: BioBERT SimCPSR Dense Retrieval (Kế thừa MedPRS Approach C)            │
│ • Pre-computed Vector Search (FAISS) trên 1,408 tạp chí PubMed                  │
│ • Output: Top 50 Candidate Journals (Recall@10 = 95.82%, Latency < 10ms)       │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Python Integrity Gate & Hard Constraint Desk Reject Engine             │
│ • Chạy 100% logic Python thuần (0ms LLM Cost):                                  │
│   1. Major Domain Matching (9 Cột nhị phân trong journal_full_info.csv)         │
│   2. Journal Integrity Gate: Check Predatory/DOAJ / Active Index                │
│   3. Desk Reject Engine (Regex scan Aims/Scope: Cấm Case Report, Cell Line...)  │
│   4. Lọc tiêu chí tác giả (Quartile Q1-Q4 / SJR threshold)                     │
│ • Output: Top 15 - 20 Candidate sạch & an toàn                                 │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: DeAR Dual-Stage Reranking & Decision-Frozen Reasoning Trace            │
│ • Sub-stage 3.1 (DeAR Pointwise Fast Scorer):                                   │
│   - Model Student 3B/8B (lora_pointwise) chấm điểm nhanh 15-20 candidates        │
│   - Thu hẹp xuống Top 10 (~0.3s)                                                │
│ • Sub-stage 3.2 (DeAR Listwise CoT Reranker & Reasoning Trace):                 │
│   - Model Student 3B/8B (lora_listwise) chạy Single-Pass Listwise CoT           │
│   - Áp dụng Scope-Over-Prestige Prompting (ưu tiên Scope fit hơn SJR/H-index)   │
│   - Sinh Reasoning Trace 2 chiều (Lý do chọn Top 1, lý do bác bỏ Top 2)         │
│   - Tự động gắn đường dẫn URL PubMed & SCImago từ journal_full_info.csv         │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
   Output Final: Top 5 Tạp chí + Full Reasoning Trace + Integrity PASS (Latency < 2.5s)
```

---

## 2. Bảng Phân Vai Các Mô-đun Trong Kiến Trúc Hợp Nhất

| Stage | Mô-đun | Vai trò cụ thể | Nguồn kỹ thuật kế thừa |
|---|---|---|---|
| **0** | **Single-Pass Parser** | Trích xuất PICO, Study Type và cờ cấm nhị phân (`is_case_report`...) vào JSON | Prompting chuẩn y học |
| **1** | **Dense Retrieval** | Encode paper query $\mathbf{e}_{paper}$, truy vấn FAISS index 1,408 journals $\rightarrow$ Top 50 | MedPRS Approach C (BioBERT SimCPSR) |
| **2** | **Integrity & Policy Gate** | Python Rule Engine (0ms LLM): Check Predatory/DOAJ, check 9 binary domains, Regex scan `Aims/Scope` loại bài báo cấm | DTAR Integrity Gate + Python Regex Engine |
| **3.1**| **DeAR Pointwise Scorer** | Model Student 3B/8B (`lora_pointwise`) chấm điểm độc lập 15-20 candidate $\rightarrow$ Top 10 (~0.3s) | DeAR Stage 1 (Distilled Pointwise) |
| **3.2**| **DeAR Listwise CoT Reranker**| Model Student 3B/8B (`lora_listwise`) chạy Listwise CoT, xuất Top 5 + Reasoning Trace + Links PubMed/SCImago (~1.5s) | DeAR Stage 2 (Listwise CoT Reasoning) |

---

## 3. Ràng buộc Latency Sau Khi Hợp Nhất (Tổng < 2.5 Giây)

| Stage | Thời gian thực thi | Cơ chế tối ưu |
|---|---|---|
| **Stage 0: Parsing** | ~ 0.5s | 1 Call LLM nhỏ duy nhất (Qwen-2.5-7B / BioBERT-NER) |
| **Stage 1: Retrieval** | < 0.01s (10ms) | FAISS Index pre-computed trên 1,408 journal embeddings |
| **Stage 2: Integrity & Rules**| 0.00s | Python Regex / Set matching thuần (Không tốn LLM) |
| **Stage 3.1: Pointwise Scorer**| ~ 0.3s | DeAR Stage 1 Pointwise LoRA Adapter (Batch evaluation) |
| **Stage 3.2: Listwise CoT** | ~ 1.5s | Single-Pass Listwise CoT Prompting (Max 400 tokens output) |
| **TỔNG CỘNG** | **~ 2.3 giây** | **Siêu nhanh, dư buffer cho Web API Production** |

---

## 4. Tóm Tắt Giá Trị Nổi Bật Của DTAR-Slim Hợp Nhất

1. **Hiệu năng Retrieval Đã Kiểm Chứng**: Kế thừa 100% khả năng lấy candidate cực mạnh từ **MedPRS Approach C** (Top-10 Accuracy 95.82%).
2. **Loại Bỏ Phức Tạp Vô Ích**: Loại bỏ hoàn toàn TourRank 10-round cồng kềnh (giảm độ trễ từ 45s xuống 2.3s, giảm chi phí từ 30 API calls xuống đúng 2 API calls).
3. **An Toàn Xuất Bản**: Tích hợp sẵn `Journal Integrity Gate` để cảnh báo tạp chí rủi ro (Predatory / Delisted) bằng Python Rule Engine 0ms.
4. **Giải Thích Tường Minh (Reasoning Trace)**: Cung cấp lý giải 2 chiều chuẩn y khoa theo DeAR CoT, giúp tác giả hiểu rõ vì sao tạp chí Top 1 phù hợp nhất với PICO bài báo của họ.
