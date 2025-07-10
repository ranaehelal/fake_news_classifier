```
fake_news_classifier/
│
├── data/
│   ├── raw/ ✅ 
│   │   ├── kaggle/ ✅ 
│   │   │   ├── Fake.csv ✅ 
│   │   │   └── True.csv ✅ 
│   │   └── fakenewsnet/ ✅ 
│   │       ├── gossip/ ✅ 
│   │       └── political/ ✅ 
│   └── processed/
│       ├── kaggle_clean.csv   ✅        # Cleaned Kaggle dataset
│       ├── fakenewsnet_clean.csv X      # Cleaned FakeNewsNet dataset
│       └── clean_data.csv      X        # Combined dataset (Kaggle + FNN)
│
├── notebooks/                          # Jupyter Notebooks for analysis and experimentation
│   ├── 01_eda.ipynb      ✅             # Exploratory Data Analysis
│   ├── 02_model_training.ipynb    ⏳     # Classical ML models (TF-IDF + Logistic, SVM)
│   ├── 03_deep_learning_model.ipynb   ⏳ # LSTM or Transformer-based models
│   └── 04_model_comparison.ipynb       # Metrics comparison, visualizations, summary
│
├── src/                                # Core source code
│   ├── data_loader.py    	✅           # Load and merge raw datasets
│   ├── preprocessing.py     ✅          # Text cleaning and normalization
│   ├── features.py           ✅         # TF-IDF vectorization, embeddings, etc.
│   ├── models/                 ⏳        # Model-specific training scripts
│   │   ├── logistic_model.py ✅ 
│   │   ├── svm_model.py X
│   │   ├── lstm_model.py  ✅ 
│   │   └── bert_model.pyX
│   ├── evaluate.py                     # Model evaluation (accuracy, F1, confusion matrix)
|   ├── eda_utils.py            ✅         
│   └── utils.py                ⏳       # Save/load models, helper functions
│
├── experiments/                        # Tracking model results
│   ├── results.csv    ✅                 # Table of metrics for all models
│   └── comparison_plot.png             # Bar chart comparing model performance
│
├── models/                             # Saved model files
│   ├── .pkl       ✅

│   ├── svm_model.pkl
│   ├── lstm_model.h5 ✅ 
│   └── bert_model.pt
│
├── visualizations/                     # All plots and graphs
│   ├── wordclouds/ ✅ 
│   │
│   └── confusion_matrices/  ✅
│       ├── logistic_cm.png ✅
│       ├── svm_cm.png   
│       └── lstm_cm.png  ✅
│
├── app/                                # Optional: Interactive demo
│   └── app.py                          # Streamlit or Gradio app
│
├── README.md                           # Project summary and instructions
├── requirements.txt      ⏳             # List of required packages
└── .gitignore                          # Files/folders to exclude from Git
```
