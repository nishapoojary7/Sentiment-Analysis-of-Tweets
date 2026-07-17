import os
import sys
import json
import uuid
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# Set UTF-8 encoding for stdout to prevent Windows console encoding errors
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import custom modules
from predict import SentimentPredictor
from database import HistoryDatabase

app = Flask(__name__)

# Configure upload folder
workspace_dir = os.path.dirname(os.path.abspath(__file__))
upload_dir = os.path.join(workspace_dir, "uploads")
static_dir = os.path.join(workspace_dir, "static")
os.makedirs(upload_dir, exist_ok=True)

app.config['UPLOAD_FOLDER'] = upload_dir
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['SECRET_KEY'] = 'sentiment-analysis-secret-key-12345'

# Initialize predictor and database
predictor = SentimentPredictor()
db = HistoryDatabase()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Retrieve tweet text from form or json
    tweet = None
    is_ajax = False
    
    if request.is_json:
        data = request.get_json()
        tweet = data.get('tweet')
        is_ajax = True
    elif 'tweet' in request.form:
        tweet = request.form['tweet']
        # Check if requested via AJAX or standard form submit
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Accept') == 'application/json':
            is_ajax = True

    if not tweet or not tweet.strip():
        if is_ajax:
            return jsonify({"error": "Empty tweet text provided."}), 400
        else:
            return redirect(url_for('index'))

    # Run Prediction
    result = predictor.predict(tweet)
    
    # Save to SQLite Database prediction history
    db.save_prediction(
        original_text=tweet,
        clean_text=result['clean_text'],
        sentiment=result['sentiment'],
        confidence=result['confidence']
    )

    if is_ajax:
        return jsonify(result)
    else:
        # Standard form post fallback: render templates/result.html
        return render_template(
            'result.html',
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            original_text=tweet,
            clean_text=result['clean_text']
        )

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles CSV file upload, processes each row, saves predictions, and returns summary stats.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Please upload a CSV file."}), 400
        
    try:
        # Save file securely with random UUID to prevent collisions
        original_name = secure_filename(file.filename)
        unique_prefix = str(uuid.uuid4())[:8]
        filename = f"{unique_prefix}_{original_name}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read CSV file using Pandas
        df = pd.read_csv(file_path)
        
        # Look for the tweet text column (case insensitive match)
        tweet_col = None
        for col in df.columns:
            if col.lower() in ['tweet', 'text', 'tweets', 'tweet_text', 'content', 'body']:
                tweet_col = col
                break
                
        if not tweet_col:
            # Fallback to the first column if no match found
            tweet_col = df.columns[0]
            
        print(f"Using column '{tweet_col}' for sentiment analysis.")
        
        results = []
        df_sentiments = []
        df_confidences = []
        
        # Process each row
        for idx, row in df.iterrows():
            tweet_text = str(row[tweet_col])
            
            # Predict
            pred = predictor.predict(tweet_text)
            
            # Save prediction to SQLite database
            db.save_prediction(
                original_text=tweet_text,
                clean_text=pred['clean_text'],
                sentiment=pred['sentiment'],
                confidence=pred['confidence']
            )
            
            # Append results for CSV annotation
            df_sentiments.append(pred['sentiment'])
            df_confidences.append(pred['confidence'])
            
            # Record for JSON response (limit display to first 100 rows for size limits)
            if len(results) < 100:
                results.append({
                    "original_tweet": tweet_text,
                    "sentiment": pred['sentiment'],
                    "confidence": pred['confidence']
                })
                
        # Annotate CSV and save back
        df['Predicted_Sentiment'] = df_sentiments
        df['Confidence_Score'] = df_confidences
        
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        df.to_csv(annotated_path, index=False)
        
        return jsonify({
            "success": True,
            "predictions": results,
            "total_rows": len(df),
            "output_file": annotated_filename
        })
        
    except Exception as e:
        print(f"Error processing upload file: {e}")
        return jsonify({"error": f"Error parsing CSV: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def download_file(filename):
    """
    Serves the annotated CSV files for user download.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/dashboard-stats', methods=['GET'])
def dashboard_stats():
    """
    Returns real-time sqlite database statistics, history logs, and training performance metadata.
    """
    sentiment_filter = request.args.get('sentiment')
    search_query = request.args.get('search')
    
    # 1. Get database records history
    history = db.get_history(limit=100, sentiment_filter=sentiment_filter, search_query=search_query)
    
    # 2. Get database statistics counts
    db_stats = db.get_stats()
    
    # 3. Load model training performance metadata
    model_metadata = None
    metadata_path = os.path.join(static_dir, "models_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        except Exception as e:
            print(f"Error loading models metadata JSON: {e}")
            
    return jsonify({
        "db_stats": db_stats,
        "history": history,
        "model_metadata": model_metadata
    })

@app.route('/delete/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    """
    Deletes a specific history record.
    """
    success = db.delete_prediction(prediction_id)
    return jsonify({"success": success})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """
    Clears all history logs.
    """
    success = db.clear_history()
    return jsonify({"success": success})

if __name__ == '__main__':
    # Print launch confirmation
    print("SentiTweet server starting at http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
