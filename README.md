## 📰 Fake News Classifier

This project is an **end-to-end Fake News Detection System** powered by:

-   ✅ Machine Learning (Logistic Regression with TF-IDF)
    
-   ✅ Deep Learning (LSTM with Embedding + padding)
    
-   ✅ 🔌 FastAPI backend
    
-   ✅ 📱 Flutter mobile app frontend
    

----------

### 📁 Project Structure

```
fake_news_classifier/
├── notebooks/                  # EDA + model training
├── src/                        # Core backend (ML/DL, API, utils)
├── models/                     # Saved models (.pkl, .h5)
├── experiments/                # Metrics + visualizations
├── requirements.txt            # Python dependencies
├── README.md                   # 🚀 You're here!
└── .gitignore
🔀 Branches:
├── main/                       # Backend: FastAPI + ML/DL models
└── flutter_app/                # Frontend: Flutter mobile app

```

----------

## 🔍 Problem Statement

Fake news has become a widespread issue online. This system detects whether a given news statement is **REAL** or **FAKE**, using both traditional ML and Deep Learning approaches.

----------

## 🚀 Features

### ✅ Backend (FastAPI)

-   `/predict`: Accepts POST request with `text` and `model` name.
    
-   Two models supported:
    
    -   `"logistic"` (TF-IDF + Logistic Regression)
        
    -   `"lstm"` (Embedding + LSTM)
        
-   Text is cleaned automatically before prediction.
    

### 🧠 Models

-   `logistic_model.pkl`: Trained on TF-IDF features
    
-   `lstm_model.h5`: Trained using padded sequences + embedding
    
-   Input preprocessing handled by:
    
    -   `clean_for_ml()` for logistic
        
    -   `clean_for_dl()` for LSTM
        

----------

## 🧪 API Usage

**📫 Endpoint:** `POST /predict`  
**📄 Request Body:**

```json
{
  "text": "The president announced a new economic policy...",
  "model": "logistic"
}

```

**✅ Response:**

```json
{
  "prediction": "real"
}
```

Use Postman or Curl to test:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"Some news text here", "model":"lstm"}'

```

----------

## 📱 Flutter App

A simple Flutter UI that sends a text to your FastAPI endpoint and displays the predicted result.

### ✨ Features

-   Clean input form
    
-   Real-time prediction on click
    
-   Supports:
    
    -   ✅ "Loading" spinner
        
    -   ✅ Error handling
        
    -   ✅ Result styling (`FAKE` in red, `REAL` in green)
        

### 🧩 Flutter Setup

1.  Clone or copy the `Flutter` project
    
2.  Replace `http://127.0.0.1:8000` with `http://10.0.2.2:8000` if using an **Android emulator**
    
3.  Run via Android Studio or command line:
    

```bash
flutter pub get
flutter run

```

----------

## 📷 Screenshots

Prediction: REAL
![Real](UI/true.png)

----------

Prediction: FAKE
![Fake](UI/fake.png)

----------

Loading
![Fake](UI/loading.png)


----------

## 📦 Dependencies

### Backend

```bash
# Environment setup
conda create -n fake_news_env python=3.10
conda activate fake_news_env

# Install packages
pip install -r requirements.txt

```

----------


## Author

 by **Rana Helal**  
[🔗 GitHub](https://github.com/ranaehelal)

----------
