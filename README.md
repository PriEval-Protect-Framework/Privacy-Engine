# PriEval-Protect: Privacy Metrics Engine

This repository contains the **Privacy Metrics Engine** of the [PriEval-Protect Framework](https://github.com/PriEval-Protect-Framework), responsible for computing data-level privacy risks using a suite of statistical, probabilistic, and information-theoretic metrics.

## Overview

This module performs **privacy risk evaluation** on structured datasets, identifying potential re-identification threats, uncertainty, and data disclosure risks. It supports automated classification of dataset attributes into:

- Quasi-identifiers (QIDs)
- Sensitive Attributes (SAs)
- Non-sensitive attributes (NSAs)

It is designed for use in e-health privacy audits and compliance scoring pipelines.

## Key Features

- Attribute classification using algorithmic and LLM-based techniques  
- Risk estimation: k-anonymity, l-diversity, entropy, mutual information  
- Adversary success likelihood & delta presence analysis  
- Modular utils for integration with scoring or compliance modules  
- FastAPI-compatible

## Project Structure

```bash
Privacy-Engine/
├── data/
│   └── test/                            # Sample test datasets
│
├── src/
│   ├── utils/                           # Core metric calculators
│   │   ├── adversary_success.py         # Adversary success and delta presence
│   │   ├── algorithmic_attribute_classification.py
│   │   ├── data_similarity.py           # k-anonymity, l-diversity
│   │   ├── info_gain_loss.py            # Mutual information, entropy
│   │   ├── llm_attribute_classification.py
│   │   └── uncertainty.py               # Entropy metrics
│   └── main.py                   # Entry point for metric pipeline
│
├── .env.example                  # Environment variable template
├── .gitignore
├── README.md
└── requirements.txt              # Python dependencies
````

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/PriEval-Protect-Framework/Privacy-Engine.git
cd Privacy-Engine
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run :

```bash
uvicorn src.main:app --reload 
```

## Privacy Metrics Computed

* **k-Anonymity**, **l-Diversity**, **α-k Anonymity**
* **Mutual Information**, **Conditional Entropy**
* **Delta Presence**, **Adversary Success Rate**
* **Normalized Shannon Entropy**
* **Risk-Based Attribute Classification** (QID, SA, NSA)

## License

This module is licensed under the [MIT License](./LICENSE).

## Authors

Developed by **Ilef Chebil** and **Asma ElHadj**
Supervised by \[EFREI Paris] and \[INSAT Tunisia]

## Related Modules

* [Compliance Assessment](https://github.com/PriEval-Protect-Framework/Compliance-Assessment)
* [Dashboard UI](https://github.com/PriEval-Protect-Framework/Dashboard-UI)
