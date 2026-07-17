# SentiTweet: Tweet Sentiment Analysis Dashboard

An attractive, interactive, and modernistic web dashboard for classifying Twitter sentiments using Machine Learning and Natural Language Processing. The project compares multiple classification algorithms and saves the best performer for real-time predictions.

---

## 🌟 Features

- **Real-Time Tweet Analyzer**: Type or paste any tweet to evaluate its sentiment (Positive, Neutral, Negative) along with confidence scores, probability gauge bars, and preprocessed NLP tokens.
- **Batch CSV Processor**: Drag-and-drop file uploader to classify thousands of tweets in bulk, view predictions in a structured paginated table, and download the annotated CSV.
- **Interactive Visual Dashboard**:
  - Donut chart showing database sentiment distribution.
  - Performance benchmarks chart comparing Logistic Regression, Multinomial Naive Bayes, Decision Trees, and Support Vector Machines.
  - Interactive confusion matrix heatmap.
  - Sentiment vocabulary frequency lists with dynamic percentage bars.
  - Static Word Clouds for positive and negative sentiments.
- **History Logs Database**: A persistent SQLite log table showing historical classification results with real-time text search, sentiment filters, and record deletion.

---

## 📂 Project Structure

```
Sentiment-Analysis/
│
├── app.py                     # Main Flask web application server
├── train_model.py             # Script to preprocess, train, compare, and save ML models
├── predict.py                 # Predictor utility class for single & batch prediction
├── database.py                # Database helper (SQLite) for history storage
├── generate_dataset.py        # Helper to generate a realistic Twitter dataset
├── requirements.txt           # Python dependency file
├── README.md                  # Project documentation
│
├── dataset/
│     └── tweets.csv           # Training dataset
│
├── models/
│     ├── sentiment_model.pkl  # Pickled best-performing ML model (Logistic Regression)
│     ├── vectorizer.pkl       # Fitted TF-IDF vectorizer
│     └── history.db           # SQLite database for storing history
│
├── preprocessing/
│     └── clean_text.py        # Advanced text cleaning and NLP preprocessing (lemmatizer)
│
├── static/
│     ├── css/
│     │    └── style.css       # Custom modern glassmorphism styling
│     ├── js/
│     │    └── main.js         # Frontend interactive logic and Chart.js controllers
│     ├── images/              # Generated word clouds and matplotlib reports
│     └── models_metadata.json # Exported training performance metrics JSON
│
└── templates/
      ├── index.html           # Beautiful responsive SPA-like dashboard UI
      └── result.html          # Standard fallback result page
```

---

## 📊 Model Training Results

The training pipeline evaluates several classifiers on the dataset using a TF-IDF vectorizer (`ngram_range=(1,2)`, `max_features=5000`) and an 80/20 train/test split:

| Model | Accuracy | F1-Score | Status |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **100%** | **1.00** | **Selected & Saved** |
| Multinomial Naive Bayes | 100% | 1.00 | Benchmarked |
| Decision Tree Classifier | 100% | 1.00 | Benchmarked |
| Support Vector Machine (SVC) | 100% | 1.00 | Benchmarked |

*Note: The high accuracy is achieved due to the distinct features of the generated training corpus, making it perfect for layout demonstration.*

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.10 or higher installed.

### 1. Install Dependencies
Navigate to the project root directory and run:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
If `dataset/tweets.csv` is not present, generate it by running:
```bash
python generate_dataset.py
```

### 3. Run the Training Pipeline
Preprocess the tweets, train models, compare metrics, and generate report assets:
```bash
python train_model.py
```

### 4. Start the Web Server
Launch the Flask development server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
Enjoy the beautiful dark-themed SentiTweet Dashboard!
