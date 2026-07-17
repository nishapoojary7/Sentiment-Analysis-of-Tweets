import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Set UTF-8 encoding for stdout to prevent Windows console encoding errors
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import clean_text
from preprocessing.clean_text import preprocess_text

# Define paths
workspace_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(workspace_dir, "dataset", "tweets.csv")
models_dir = os.path.join(workspace_dir, "models")
reports_dir = os.path.join(workspace_dir, "reports")
static_dir = os.path.join(workspace_dir, "static")
static_images_dir = os.path.join(static_dir, "images")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(static_images_dir, exist_ok=True)

def extract_top_words(df, sentiment, top_n=15):
    """
    Extracts the most frequent words in tweets belonging to a specific sentiment.
    """
    from collections import Counter
    text_data = " ".join(df[df['Sentiment'] == sentiment]['Cleaned_Tweet'].astype(str))
    words = text_data.split()
    # Filter out empty or single-character words
    words = [w for w in words if len(w) > 1]
    counter = Counter(words)
    return [{"word": word, "count": count} for word, count in counter.most_common(top_n)]

def generate_word_cloud(df, sentiment, filename):
    """
    Generates a word cloud image for the given sentiment and saves it.
    """
    try:
        from wordcloud import WordCloud
        text_data = " ".join(df[df['Sentiment'] == sentiment]['Cleaned_Tweet'].astype(str))
        if not text_data.strip():
            return False
        
        # Select colormap based on sentiment
        if sentiment == "Positive":
            colormap = 'viridis'
            bg_color = 'black'
        elif sentiment == "Negative":
            colormap = 'magma'
            bg_color = 'black'
        else:
            colormap = 'cool'
            bg_color = 'black'
            
        wc = WordCloud(width=800, height=400, background_color=bg_color, colormap=colormap, max_words=100).generate(text_data)
        
        # Save to reports and static/images
        wc.to_file(os.path.join(reports_dir, f"wordcloud_{sentiment.lower()}.png"))
        wc.to_file(os.path.join(static_images_dir, f"wordcloud_{sentiment.lower()}.png"))
        return True
    except Exception as e:
        print(f"Warning: WordCloud generation failed for {sentiment}: {e}")
        return False

def main():
    print("Step 2 & 3: Loading and Inspecting Dataset...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run generate_dataset.py first.")
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates count: {df.duplicated().sum()}")
    
    # Remove duplicates and missing values if any
    df = df.dropna().drop_duplicates()
    print(f"Dataset shape after cleaning duplicates: {df.shape}")
    
    print("\nStep 4: Preprocessing Tweets (this may take a few seconds)...")
    df['Cleaned_Tweet'] = df['Tweet'].apply(lambda x: preprocess_text(x, method='lemmatize'))
    
    # Filter out rows where Cleaned_Tweet is empty after preprocessing
    df = df[df['Cleaned_Tweet'].str.strip() != ""]
    print(f"Dataset shape after empty preprocessed removal: {df.shape}")
    
    # Sentiment distribution statistics
    dist = df['Sentiment'].value_counts().to_dict()
    print(f"Sentiment distribution: {dist}")
    
    print("\nStep 5: EDA and Word Cloud generation...")
    # Generate Word Clouds
    generate_word_cloud(df, "Positive", "wordcloud_positive.png")
    generate_word_cloud(df, "Negative", "wordcloud_negative.png")
    generate_word_cloud(df, "Neutral", "wordcloud_neutral.png")
    
    # Extract top words for positive/negative/neutral
    top_pos_words = extract_top_words(df, "Positive", 15)
    top_neg_words = extract_top_words(df, "Negative", 15)
    top_neut_words = extract_top_words(df, "Neutral", 15)
    
    print("\nStep 6: Feature Extraction (TF-IDF)...")
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(df['Cleaned_Tweet'])
    y = df['Sentiment']
    
    print("\nStep 7: Splitting Dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    print("\nStep 8: Training Machine Learning Models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_accuracy = -1.0
    best_model = None
    
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
        
        print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = clf
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    print("\nStep 9: Evaluating Best Model...")
    # Best model evaluation
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=['Negative', 'Neutral', 'Positive'])
    
    # Save best model and vectorizer
    print("\nStep 10: Saving Best Model and Vectorizer...")
    with open(os.path.join(models_dir, "sentiment_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
        
    with open(os.path.join(models_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Model and Vectorizer saved successfully.")
    
    # Export metrics JSON
    metadata = {
        "best_model": best_model_name,
        "best_accuracy": best_accuracy,
        "models_comparison": results,
        "confusion_matrix": {
            "labels": ["Negative", "Neutral", "Positive"],
            "matrix": cm.tolist()
        },
        "top_words": {
            "Positive": top_pos_words,
            "Negative": top_neg_words,
            "Neutral": top_neut_words
        },
        "dataset_stats": {
            "total_tweets": len(df),
            "distribution": dist
        }
    }
    
    metadata_path = os.path.join(static_dir, "models_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metrics metadata written to: {metadata_path}")
    
    # Generate Visualizations for Reports
    # 1. Model accuracy comparison bar chart
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    f1_scores = [results[n]["f1_score"] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#007BFF')
    ax.bar(x + width/2, f1_scores, width, label='F1-Score (Macro)', color='#28A745')
    
    ax.set_ylabel('Scores')
    ax.set_title('Machine Learning Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(reports_dir, "model_comparison.png"), dpi=150)
    plt.savefig(os.path.join(static_images_dir, "model_comparison.png"), dpi=150)
    plt.close()
    
    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=150)
    plt.savefig(os.path.join(static_images_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    print("Training pipeline completed and report images generated.")

if __name__ == "__main__":
    main()
