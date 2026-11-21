# Existing patent retrieval methods and the research gap

> Grounded in the 2017 survey (Shalaby & Zadrozny) and recent transformer-based patent intelligence literature.

## 1. Existing methods & Known Limitations
| Family | Idea | Reported limits (from literature & surveys) |
|---|---|---|
| Query reformulation | Pick/weight terms from the topic patent (claims, description, IPC filters) | Long queries; best CLEF-IP 2010 MAP about 0.12 |
| Pseudo-relevance feedback | Expand/reduce the query with top-ranked docs | Parameter-sensitive, unstable; query reduction often beats expansion |
| Thesaurus / semantic expansion | WordNet, query-log, Wikipedia, topic models | Small gains; WordNet raised MAP slightly but lowered PRES and slowed queries |
| Citation / classification (IPC, CPC) | PageRank on citations, class vectors | Incomplete citation graphs; class revisions |
| Interactive search | Faceted / visual refinement | Not automatic; hard to evaluate |
| Dense / neural (later work) | Patent-tuned Sentence-BERT-style encoders, paragraph aggregation (e.g. PARM), embedding benchmarks | Long documents exceed encoder input length; patent legal syntax; character truncation artifacts |

Survey conclusions: automatic performance is historically low (<0.2 MAP); ~12% of relevant CLEF-IP documents share no terms with their topic patent (vocabulary mismatch); citation graphs bias towards related work rather than pure 35 U.S.C. § 102/103 patentability anticipation.

## 2. Research Gaps This Project Formally Solves
1. **Lexical vs Dense Semantic vs Neural Hybrid under Controlled Split**: Direct comparison of Okapi BM25, SBERT/PatentSBERTa, Calibrated Score Fusion, RRF, and Cross-Encoder re-ranking on identical patent benchmarks.
2. **Vocabulary Mismatch Quantification via VMRR**: Formally measuring the **Vocabulary Mismatch Recovery Rate (VMRR@K)**: the fraction of relevant prior-art patents sharing zero lexical terms that are successfully retrieved by dense neural components.
3. **From Document-Level Similarity to Clause-Level Anticipation (35 U.S.C. § 102/103)**: Moving beyond coarse bag-of-words similarity by introducing **automated Element-by-Element Claim Charting**, decomposing claims into clauses and calculating clause-level anticipation against prior-art disclosures.
4. **Token-Aware Context Windowing**: Eliminating naive character truncation to ensure full claim text is evaluated during neural cross-attention re-ranking.
5. **Statistical Significance Testing**: Validating whether hybrid improvements over lexical baselines are statistically significant using two-sided **Wilcoxon Signed-Rank tests** ($p < 0.05, p < 0.01$).
6. **Hard-Negative Contrastive Adaptation**: Mining BM25 false-positive citations as hard negatives for domain fine-tuning under `MultipleNegativesRankingLoss`.

## 3. How the Codebase Implements This
- `search_engine.py`: Implements BM25 with patent stopword filtration, dense embeddings with disk caching, RRF/min-max score calibration, word-aware Cross-Encoder re-ranking, and automated 35 U.S.C. § 102/103 Claim Limitation decomposition.
- `evaluator.py`: Computes MAP, MRR, nDCG@K, Precision@K, Recall@K, VMRR@10 (%), and paired Wilcoxon statistical significance p-values.
- `trainer.py`: PyTorch domain-adaptive fine-tuning with live step loss logging and BM25 hard-negative mining.
- `app.py`: Streamlit prior-art assessment studio displaying the interactive Claim Chart, novelty collision risk gauge, 2D semantic landscape, and 1-click formal examination report generator.

## Sources
- Shalaby & Zadrozny, [Patent Retrieval: A Literature Review](https://arxiv.org/pdf/1701.00324)
- [PARM: Paragraph Aggregation Retrieval Model](https://arxiv.org/pdf/2201.01614)
- [Benchmarking Patent Embeddings: 22 Models](https://arxiv.org/html/2605.24297)
- [DAPFAM patent retrieval dataset](https://www.researchgate.net/publication/393148782_DAPFAM_A_Domain-Aware_Patent_Retrieval_Dataset_Aggregated_at_the_Family_Level)
- [NLP in the Patent Domain: A Survey](https://arxiv.org/pdf/2403.04105)
