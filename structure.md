```
fake_news_classifier/
│
├── data/
│   ├── raw/                        # Raw datasets
│   │   ├── kaggle/
│   │   │   ├── Fake.csv ✅
│   │   │   └── True.csv ✅
│   │   └── fakenewsnet/
│   │       ├── gossip/ ✅
│   │       └── political/ ✅
│   └── processed/                 # Cleaned and combined datasets
│       ├── kaggle_clean.csv ✅
│       ├── fakenewsnet_clean.csv ✖️
│       └── clean_data.csv ✖️
│
├── notebooks/                     # Jupyter notebooks for experiments
│   ├── 01_eda.ipynb ✅             # Exploratory Data Analysis
│   ├── 02_model_training.ipynb ✅  # TF-IDF + Logistic, SVM
│   ├── 03_deep_learning_model.ipynb ✅  # LSTM or Transformer models
│   └── 04_model_comparison.ipynb ✖️     # Metrics, visualizations, summary
│
├── src/                           # Core source code
│   ├── data_loader.py ✅           # Load and merge datasets
│   ├── preprocessing.py ✅         # Text cleaning
│   ├── features.py ✅              # TF-IDF, embeddings
│   ├── evaluate.py ✅              # Evaluation metrics
│   ├── run_pipline.py ✅           # Main pipeline runner
│   ├── predict_text.py ✅          # Predict from text input
│   ├── utils/
│   │   ├── eda_utils.py ✅
│   │   └── utils.py ✅             # Save/load models, helpers
│   └── models/                  # Model training scripts
│       ├── logistic_model.py ✅
│       ├── svm_model.py ✖️
│       ├── lstm_model.py ✅
│       └── bert_model.py ✖️
│
├── experiments/                   # Results tracking
│   ├── results.csv ✅              # All model metrics
│   └── comparison_plot.png ✖️     # Bar chart of model performance
│
├── models/                        # Saved model files
│   ├── logistic_model.pkl ✅
│   ├── svm_model.pkl ✖️
│   ├── lstm_model.h5 ✅
│   └── bert_model.pt ✖️
│
├── visualizations/                # Plots and graphs
│   ├── wordclouds/ ✅
│   └── confusion_matrices/
│       ├── logistic_cm.png ✅
│       ├── svm_cm.png ✖️
│       └── lstm_cm.png ✅
│
├── app/                           # Optional UI (Streamlit or Gradio)
│   └── app.py
│
├── UI/                            # Static assets (for README)
│   ├── true.png
│   ├── fake.png
│   └── screenshots/               # Flutter app screenshots
│
├── README.md ✅                   # Project overview
├── requirements.txt ✅            # Python dependencies
└── .gitignore ✅                  # Git ignored files
```
