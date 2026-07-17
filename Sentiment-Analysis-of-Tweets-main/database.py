import os
import sqlite3
from datetime import datetime

class HistoryDatabase:
    def __init__(self):
        self.workspace_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.workspace_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.db_path = os.path.join(self.models_dir, "history.db")
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """
        Creates the history table if it doesn't already exist.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                clean_text TEXT,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_prediction(self, original_text, clean_text, sentiment, confidence):
        """
        Saves a prediction entry to the database.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prediction_history (original_text, clean_text, sentiment, confidence)
                VALUES (?, ?, ?, ?)
            ''', (original_text, clean_text, sentiment, confidence))
            
            conn.commit()
            last_id = cursor.lastrowid
            conn.close()
            return last_id
        except Exception as e:
            print(f"Error saving prediction to database: {e}")
            return None

    def get_history(self, limit=100, sentiment_filter=None, search_query=None):
        """
        Retrieves historical prediction records, optionally filtered or searched.
        """
        try:
            conn = self.get_connection()
            conn.row_factory = sqlite3.Row  # Returns dictionaries/row objects instead of tuples
            cursor = conn.cursor()
            
            query = "SELECT * FROM prediction_history WHERE 1=1"
            params = []
            
            if sentiment_filter:
                query += " AND sentiment = ?"
                params.append(sentiment_filter)
                
            if search_query:
                query += " AND (original_text LIKE ? OR clean_text LIKE ?)"
                params.extend([f"%{search_query}%", f"%{search_query}%"])
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "id": row["id"],
                    "original_text": row["original_text"],
                    "clean_text": row["clean_text"],
                    "sentiment": row["sentiment"],
                    "confidence": row["confidence"],
                    "timestamp": row["timestamp"]
                })
                
            conn.close()
            return history
        except Exception as e:
            print(f"Error getting history: {e}")
            return []

    def delete_prediction(self, prediction_id):
        """
        Deletes a specific prediction by ID.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_history WHERE id = ?", (prediction_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting prediction {prediction_id}: {e}")
            return False

    def clear_history(self):
        """
        Deletes all history records.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_history")
            # Reset autoincrement sequence
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='prediction_history'")
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False

    def get_stats(self):
        """
        Calculates basic stats from history: counts by sentiment, total analyzed.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM prediction_history")
            total = cursor.fetchone()[0]
            
            # Counts by sentiment
            cursor.execute("SELECT sentiment, COUNT(*) FROM prediction_history GROUP BY sentiment")
            rows = cursor.fetchall()
            
            distribution = {"Positive": 0, "Neutral": 0, "Negative": 0}
            for row in rows:
                sentiment, count = row
                if sentiment in distribution:
                    distribution[sentiment] = count
                    
            conn.close()
            return {
                "total": total,
                "distribution": distribution
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total": 0, "distribution": {"Positive": 0, "Neutral": 0, "Negative": 0}}

if __name__ == "__main__":
    db = HistoryDatabase()
    print("Database path:", db.db_path)
    
    # Save a test prediction
    test_id = db.save_prediction(
        "I love this app so much!", 
        "love app much", 
        "Positive", 
        95.4
    )
    print(f"Saved test prediction. Row ID: {test_id}")
    
    # Get stats
    stats = db.get_stats()
    print("Stats:", stats)
    
    # Retrieve history
    history = db.get_history(limit=5)
    print("History:")
    for row in history:
        print(f"- [{row['timestamp']}] {row['sentiment']} ({row['confidence']}%): {row['original_text']}")
        
    # Clean up test
    db.delete_prediction(test_id)
    print("Deleted test prediction. Active stats:", db.get_stats())
