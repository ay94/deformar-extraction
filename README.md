Install by navigating to the root directory 
pip install -e .




/experiment
│
├── fine_tuning/
│   ├── model_1/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── config.yaml
│   │   └── model_files/
│   ├── model_2/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── config.yaml
│   │   └── model_files/
│   └── ... (additional models)
│
├── extraction/
│   ├── dataset_1/
│   │   ├── model_1/
│   │   │   ├── analysis/
│   │   │   ├── reports/
│   │   │   ├── similarity/
│   │   │   ├── confusion/
│   │   │   └── train/
│   │   ├── model_2/
│   │   │   ├── analysis/
│   │   │   ├── reports/
│   │   │   ├── similarity/
│   │   │   ├── confusion/
│   │   │   └── train/
│   │   └── ... (additional models)
│   │
│   ├── dataset_2/
│   │   ├── model_1/
│   │   │   ├── analysis/
│   │   │   ├── reports/
│   │   │   ├── similarity/
│   │   │   ├── confusion/
│   │   │   └── train/
│   │   ├── model_2/
│   │   │   ├── analysis/
│   │   │   ├── reports/
│   │   │   ├── similarity/
│   │   │   ├── confusion/
│   │   │   └── train/
│   │   └── ... (additional models)
│   └── ... (additional datasets)
│
├── notebooks/
│   ├── fine_tuning_notebook.ipynb
│   ├── extraction_notebook.ipynb
│   └── ... (additional notebooks)
│
├── configs/
│   ├── fine_tuning_config.yaml
│   ├── extraction_config.yaml
│   └── ... (additional config files)
│
└── README.md
