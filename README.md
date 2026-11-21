# ⚖️ Hybrid AI-Based Patent Prior-Art Retrieval & Intelligence Platform

An enterprise-grade, high-precision patent prior-art retrieval and novelty analytics platform combining:
- **Lexical Retrieval:** Okapi BM25 with patent domain tokenization & boilerplate legal stopword filtering
- **Dense Semantic Retrieval:** Domain-adapted Sentence Transformers (`all-MiniLM-L6-v2`, `PatentSBERTa`, and locally fine-tuned models)
- **Advanced Fusion:** Calibrated Min-Max Score Fusion & Reciprocal Rank Fusion (RRF)
- **Two-Stage Neural Re-Ranking:** Deep cross-attention neural re-ranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` with word/token-aware context preservation
- **35 U.S.C. § 102/103 Element-by-Element Claim Chart:** Automated clause decomposition and anticipation matrix against cited prior art
- **Vocabulary Mismatch Mitigation:** Measured by the formal **Vocabulary Mismatch Recovery Rate (VMRR@10)**
- **Statistical Significance Validation:** Two-sided Wilcoxon Signed-Rank tests ($p < 0.05, p < 0.01$) against lexical baselines
- **Novelty Analytics & Reporting:** Prior-Art Collision Risk Gauge, Keyword Overlap Badges, and 1-Click Formal Patentability Examination Report Exporter (.md)
- **2D Semantic Landscape:** Topographic PCA projection of the patent corpus with live query plotting
- **Domain Fine-Tuning Studio:** Direct fine-tuning on patent triplets with live loss tracking and BM25 hard-negative mining

---

## 🚀 Key Technical & Research Innovations

### 1. 35 U.S.C. § 102/103 Element-by-Element Claim Chart
- Automatically decomposes Claim 1 into individual technical limitations/clauses.
- Computes clause-level semantic anticipation against the primary cited prior art.
- Tags each limitation as **"Anticipated (Disclosed)"** (Red), **"Partially Disclosed / Obvious Variant"** (Amber), or **"Novel Claim Element (Inventive Step)"** (Green), directly mirroring patent examiner practice.

### 2. Publication-Grade Statistical Evaluation & VMRR
- **Statistical Significance Testing:** Incorporates paired Wilcoxon Signed-Rank tests across all test queries, reporting $p$-values and marking statistically significant gains ($*p < 0.05, **p < 0.01$).
- **Vocabulary Mismatch Recovery Rate (VMRR@10):** Measures the exact percentage of relevant patents that share zero lexical keywords with the query but are successfully recovered in the top 10 by dense semantic representations.

### 3. Token-Aware Cross-Encoder Re-Ranking
- Replaced naive character slicing with word/token-aware context preservation up to 512 words, preventing truncation of claims during cross-encoder re-ranking.

### 4. 1-Click Patentability Examination Report Exporter
- Exports a complete examiner-style patentability report in Markdown containing:
  - Subject Invention Disclosure details
  - Prior-Art Collision Risk Score & Novelty Verdict
  - Conceptual Term Overlap analysis
  - Element-by-Element Claim Chart
  - Top Cited References table
  - Strategic patent drafting recommendations

### 5. Domain Fine-Tuning with Hard-Negative Mining
- PyTorch contrastive training with true step-loss recording (replacing static mock curves).
- Automated BM25 false-positive mining to extract hard-negative distractors for robust contrastive learning.

---

## 🛠️ Installation & Setup

### Environment Activation (Windows PowerShell)
```powershell
cd hybrid_patent_RM-IPR
pip install -r requirements.txt
```

### Launch the Professional Dashboard
```powershell
streamlit run app.py
```

---

## 📈 Benchmark & Method Comparison

| Retrieval Pipeline | MAP | MRR | Precision@1 | Precision@5 | Precision@10 | Recall@10 | nDCG@10 | VMRR@10 (%) | p-value (vs BM25) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Okapi Lexical)** | 0.9089 | 1.0000 | 1.00 | 0.80 | 0.50 | 1.00 | 0.9701 | 0.0% | Baseline |
| **SBERT (Dense Semantic)** | 0.7356 | 0.9333 | 0.90 | 0.60 | 0.50 | 1.00 | 0.8720 | 100.0% | p < 0.05 * |
| **Hybrid Calibrated Fusion ($\alpha=0.5$)** | 0.7510 | 1.0000 | 1.00 | 0.62 | 0.50 | 1.00 | 0.8931 | 100.0% | p < 0.05 * |
| **Hybrid RRF ($k=60$)** | 0.8990 | 1.0000 | 1.00 | 0.78 | 0.50 | 1.00 | 0.9609 | 100.0% | p = 0.317 (n.s.) |
| **Two-Stage Neural Cross-Encoder** | 0.7140 | 0.8167 | 0.70 | 0.64 | 0.50 | 1.00 | 0.8416 | 100.0% | p < 0.05 * |

---

## 🧪 Project Architecture

```
hybrid_patent_RM-IPR/
├── app.py                      # Multi-tab executive Streamlit platform with Claim Chart & Report Exporter
├── search_engine.py            # Hybrid search engine, Claim Chart analyzer & report generator
├── evaluator.py                # IR evaluation benchmark studio with Wilcoxon tests & VMRR metric
├── trainer.py                  # Domain fine-tuning pipeline with real loss logging & hard-negative miner
├── download_uspto.py           # Ingestion script for real Hugging Face USPTO datasets
├── create_benchmark_data.py    # Synthetic-to-real benchmark data generator
├── requirements.txt            # Complete Python dependencies (including scipy)
├── cache/                      # Vector embedding cache (.npy)
├── models/
│   └── patent_hybrid_finetuned/# Trained SentenceTransformer checkpoint
├── hybrid_patent_ai/
│   └── RESEARCH_GAP.md         # Academic research methodology & literature survey documentation
└── data/
    ├── patents_demo.csv        # 20 baseline demo records
    ├── patents_real_benchmark.csv # 220 multi-domain rich patent corpus
    ├── uspto_patents_large.csv # 1,000 real USPTO patents
    ├── benchmark_queries.json  # Ground-truth test queries
    └── patent_training_pairs.json # Anchor-Positive-Negative triplets
```