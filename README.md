CAAAPT/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   ├── frontend_config.yaml     # XGBoost hyperparameters & thresholds
│   ├── backend_config.yaml      # LLM API settings & CoT parameters
│   └── pareto_config.yaml       # Pareto frontier search config
├── data/
│   ├── raw/                     # DARPA TC E3 / TREC raw data
│   ├── processed/               # Compressed provenance graphs
│   └── knowledge_base/          # Structured + vector knowledge base
├── src/
│   ├── frontend/
│   │   ├── compress.py          # Semantic-preserving compression
│   │   ├── feature_extractor.py # N-Gram + TF-IDF extraction
│   │   └── xgboost_screener.py  # XGBoost classifier + threshold control
│   ├── backend/
│   │   ├── knowledge_base_builder.py  # Build structured + vector KB
│   │   ├── retriever.py               # Hybrid retrieval (SQL + vector)
│   │   ├── cot_prompts.py             # Four-stage CoT templates
│   │   ├── llm_client.py              # LLM API wrapper
│   │   └── attribution_analyzer.py    # RAG + CoT reasoning
│   ├── core/
│   │   ├── pipeline.py          # End-to-end CAAAPT pipeline
│   │   ├── confidence.py        # C_final comprehensive confidence
│   │   └── human_review.py      # Low-confidence sample delegation
│   └── utils/
│       ├── logger.py            # Logging utilities
│       ├── metrics.py           # F1, Recall, TacticACC, etc.
│       └── pareto.py            # Pareto boundary analysis
├── experiments/
│   ├── ablation.py              # Ablation study (w/o XGBoost/RAG/CoT)
│   ├── pareto_search.py         # Pareto frontier grid search
│   └── cost_analysis.py         # Computational & economic cost
└── scripts/
    ├── download_darpa.sh        # Download DARPA TC E3 dataset
    └── preprocess_trec.py       # Preprocess TREC dataset


Important Notice
This is a de-identified code release for Open Science purposes only.

All code and configurations provided in this repository are sanitized versions intended solely to demonstrate the core algorithmic logic of CAAAPT. The following have been removed or anonymized:

Proprietary optimization logic

Production-specific components

Sensitive prompt templates that could be used for adversarial evasion

Internal API endpoints and credentials

Benchmark-specific hardcoded paths

These artifacts are sufficient for methodology understanding and reproduction of the results described in the paper, but do not represent the complete production system.



Configuration
Key parameters can be adjusted in config/:

frontend_config.yaml: XGBoost hyperparameters and τ_front

backend_config.yaml: LLM settings, RAG parameters, and τ_back

pareto_config.yaml: Pareto frontier search grid

Datasets
Evaluation uses two public benchmarks:

DARPA TC E3 (THEIA, CADETS, Trace subsets)

TREC APT Tactic Dataset



