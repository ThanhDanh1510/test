# Master Pipeline Hợp Nhất: DTAR-Slim Architecture (MedPRS + Fast Integrity Rules + DeAR 2-Stage Reasoning)

> **Mục tiêu**: Kết hợp ưu điểm vượt trội của **MedPRS** (BioBERT Retrieval Top-10 Acc 95.82%), **Journal Integrity Gate** (Lọc tạp chí rủi ro/predatory), **Python Desk Reject Engine** (Lọc quy tắc cứng 0ms LLM) và **DeAR Dual-Stage Reranking** (Chấm điểm nhanh + Suy luận CoT Listwise).

---

## 0. Tổng quan Kiến trúc Hợp nhất (End-to-End Flow)

```
                       [Bài báo Input: Title + Abstract + Keywords]
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: Single-Pass Fast LLM Parsing & Normalization                           │
│ • 1 Call LLM/NER duy nhất: Trích xuất PICO + Study Type                        │
│   + MeSH Terms + Cờ nhị phân cấm (is_case_report, is_cell_line...) + 9 Domains  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: BioBERT SimCPSR Dense Retrieval (Kế thừa MedPRS Approach C)            │
│ • Pre-computed Vector Search (FAISS) trên 1,408 tạp chí PubMed                  │
│ • Output: Top 50 Candidate Journals (Recall@10 = 95.82%, Thời gian < 10ms)     │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Python Integrity Gate & Hard Constraint Desk Reject Engine             │
│ • Chạy 100% logic Python thuần (0ms LLM Cost):                                  │
│   1. Lọc 9 Nhóm ngành chính (Domain Binary Matching)                            │
│   2. Journal Integrity Gate: Lọc tạp chí rủi ro (Predatory/DOAJ / Delisted)     │
│   3. Desk Reject Engine (Regex scan Aims/Scope: Cấm Case Report, Cell Line...)  │
│   4. Lọc chỉ số tối thiểu tác giả yêu cầu (Best Quartile Q1-Q4 / SJR)          │
│ • Output: Top 15 - 20 Candidate sạch & an toàn                                 │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: DeAR Dual-Stage Reranking & Decision-Frozen Reasoning Trace            │
│ • Stage 3.1 (DeAR Pointwise Fast Scorer):                                       │
│   - Model Student 3B/8B (lora_pointwise) chấm điểm nhanh 15-20 candidates        │
│   - Thu hẹp xuống Top 10 (~0.3s)                                                │
│ • Stage 3.2 (DeAR Listwise CoT Reranker & Reasoning Trace):                     │
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

## 1. Chi tiết từng Giai đoạn trong Pipeline Hợp Nhất

### 📍 STAGE 0: Single-Pass Fast LLM Parsing & Normalization
- **Input**: `Title`, `Abstract`, `Keywords`.
- **Thực thi**: 1 Prompt JSON duy nhất gửi tới LLM nhỏ (Qwen-2.5-7B / BioBERT-NER):
  ```json
  {
    "pico_summary": "P: Patients with T2D; I: SGLT2 inhibitors; C: Placebo; O: Cardiovascular outcomes",
    "study_type": "Randomized Controlled Trial",
    "is_case_report": false,
    "is_cell_line_only": false,
    "is_animal_only": false,
    "mesh_categories": ["Endocrinology, Diabetes and Metabolism", "Cardiovascular Diseases"],
    "domain_flags": {"Medicine": 1, "Pharmacology, Toxicology and Pharmaceutics": 1}
  }
  ```
- **Lợi ích**: Chuẩn hóa thông tin y học siêu cô đọng, triệt tiêu trôi ngữ nghĩa (Semantic Drift).

---

### 📍 STAGE 1: BioBERT SimCPSR Dense Retrieval (Kế thừa MedPRS Approach C)
- **Input**: Cấu trúc dữ liệu bài báo từ Stage 0.
- **Thực thi**:
  - Mã hóa vector query $\mathbf{e}_{paper} = \text{BioBERT}(\text{Title + Abstract + Keywords + MeSH})$.
  - Truy vấn FAISS Index đối chiếu với 1,408 vector tạp chí đã pre-compute tĩnh trong `journal_full_info.csv`.
- **Output**: Top 50 Candidate Journals.
- **Thời gian**: `< 10 miligiây` (Recall@Top-10 đạt **95.82%**).

---

### 📍 STAGE 2: Python Integrity Gate & Hard Constraint Desk Reject Engine
- **Input**: Top 50 Candidate Journals từ Stage 1.
- **Thực thi**: Xử lý 100% bằng mã nguồn Python thuần (**0ms LLM cost**):
  1. **Major Domain Check**: Lọc trùng khớp 9 cột nhị phân (`Medicine`, `Neuroscience`, `Dentistry`...).
  2. **Journal Integrity Gate (An toàn xuất bản)**: Đối chiếu danh sách DOAJ / Beall's list / PubMed Active Index để loại bỏ các tạp chí rủi ro (Predatory / Delisted).
  3. **Desk Reject Engine**: Regex scan văn bản `Aims` & `Scope` trong CSV đối chiếu cờ nhị phân Stage 0:
     - Nếu `is_case_report == True` VÀ `Aims/Scope` ghi cấm Case Reports $\rightarrow$ Loại ngay.
     - Nếu `is_cell_line_only == True` VÀ `Aims/Scope` ghi cấm thuần Cell Line $\rightarrow$ Loại ngay.
  4. **User Preferences**: Lọc chỉ số tối thiểu do tác giả quy định (Best Quartile Q1-Q4 / SJR rank).
- **Output**: Top 15 – 20 Candidates sạch, an toàn 100%.

---

### 📍 STAGE 3: DeAR Dual-Stage Reranking & Decision-Frozen Reasoning Trace
- **Input**: Top 15 – 20 Candidates từ Stage 2.
- **Thực thi (2 Sub-stages)**:
  - **Sub-stage 3.1 (DeAR Pointwise Fast Scorer)**:
    - Sử dụng `Qwen-2.5-7B` mang LoRA Adapter `lora_pointwise` chấm điểm độc lập 15-20 candidates với thời gian siêu nhanh (~0.3s).
    - Cắt lấy Top 10 tạp chí điểm cao nhất.
  - **Sub-stage 3.2 (DeAR Listwise CoT Reranker & Reasoning Trace)**:
    - Đổi sang LoRA Adapter `lora_listwise`.
    - Gửi Top 10 candidates vào **1 Prompt Listwise duy nhất** với quy tắc **Scope-Over-Prestige Prompting** (ép LLM ưu tiên Scope fit hơn chỉ số danh tiếng SJR).
    - LLM xuất ra bảng xếp hạng Top 5 kèm **Reasoning Trace** 2 chiều (Lý do chọn Top 1, lý do đẩy Top 2 xuống vị trí thứ 2).
    - Tự động gắn liên kết `URL` (PubMed) và `URL_Scimago` từ `journal_full_info.csv`.
- **Output**: Top 5 Tạp chí + Full Reasoning Trace + Link tra cứu.

---

## 2. Bảng So sánh Pipeline Đơn lẻ vs Pipeline Hợp Nhất (DTAR-Slim)

| Tiêu chí | MedPRS Gốc | Pipeline TourRank Cũ | DTAR-Slim Hợp Nhất (Mới) |
|---|---|---|---|
| **Bản chất** | Classification (Softmax tĩnh) | Tournament RL (Rất cồng kềnh) | **Dense Retrieval + DeAR Reasoning** |
| **Độ trễ (Latency)** | < 0.1s (Chỉ cho điểm) | 30 - 60s (Chậm) | **< 2.5 giây (Siêu nhanh)** |
| **Số API Call LLM** | 0 | 20 - 30 calls | **Chính xác 2 calls / paper** |
| **An toàn xuất bản** | Không có | Lọc live cồng kềnh | **Journal Integrity Gate (Python Rule Engine)** |
| **Khả năng Giải thích** | Không có (Chỉ có điểm %) | Giải thích rời rạc | **Full Reasoning Trace 2 chiều (CoT)** |
| **Tính khả thi Deploy** | Cao nhưng thiếu giải thích | Rất thấp | **100% Sẵn sàng cho Web/API Production** |

---

## 3. Cấu trúc Output JSON Trả về cho User / Web Interface

```json
{
  "paper_summary": {
    "pico": "P: T2D Patients; I: SGLT2 inhibitors; C: Placebo; O: CV mortality reduction",
    "study_type": "Randomized Controlled Trial"
  },
  "recommendations": [
    {
      "rank": 1,
      "journal_title": "Journal of Clinical Endocrinology & Metabolism",
      "quartile": "Q1",
      "sjr_index": 2.15,
      "reasoning_trace": {
        "scope_fit": "Hoàn toàn trùng khớp với PICO nghiên cứu lâm sàng trên bệnh nhân tiểu đường tuýp 2.",
        "why_top1": "Tạp chí ưu tiên hàng đầu các bài thử nghiệm lâm sàng RCT về thuốc nội tiết chuyển hóa.",
        "integrity_status": "PASS (Scopus Q1, PubMed Active Index)"
      },
      "urls": {
        "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=%22J+Clin+Endocrinol+Metab%22...",
        "scimago": "https://www.scimagojr.com/journalsearch.php?q=..."
      }
    },
    {
      "rank": 2,
      "journal_title": "Therapeutic Advances in Neurological Disorders",
      "quartile": "Q1",
      "sjr_index": 1.436,
      "reasoning_trace": {
        "scope_fit": "Phù hợp một phần nếu bài báo tập trung vào biến chứng thần kinh do tiểu đường.",
        "why_not_top1": "Phạm vi chính của tạp chí thiên về Thần kinh học lâm sàng hơn là chuyển hóa tim mạch.",
        "integrity_status": "PASS (Scopus Q1, PubMed Active Index)"
      },
      "urls": {
        "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=%22Ther+Adv+Neurol+Disord%22...",
        "scimago": "https://www.scimagojr.com/journalsearch.php?q=..."
      }
    }
  ]
}
```
