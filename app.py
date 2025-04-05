import os
import sqlite3
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
DB_FILE = "shl_assessments.db"

@app.route('/')
def home():
    print("Home route accessed")
    return "Flask is running"

# Initialize Database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS assessments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        url TEXT,
                        remote_support TEXT,
                        adaptive_support TEXT,
                        duration TEXT,
                        test_type TEXT,
                        description TEXT)
                    ''')
    conn.commit()
    conn.close()

# Load Data into Database (Dummy Data for Now)
def load_dummy_data():
    data = [
        ("Java Developer Test", "https://shl.com/java-test", "Yes", "Yes", "40 mins", "Technical", "Test for Java developers"),
        ("Python & SQL Test", "https://shl.com/python-sql", "Yes", "No", "60 mins", "Technical", "Covers Python, SQL, JavaScript"),
        ("Cognitive & Personality Test", "https://shl.com/cognitive", "Yes", "Yes", "45 mins", "Cognitive", "Cognitive and personality assessment")
    ]
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.executemany('''INSERT INTO assessments (name, url, remote_support, adaptive_support, duration, test_type, description)
                          VALUES (?, ?, ?, ?, ?, ?, ?)''', data)
    conn.commit()
    conn.close()

# Function to Fetch Recommendations
def get_recommendations(query, top_n=10):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM assessments", conn)
    conn.close()
    
    if df.empty:
        return []
    
    # ✅ If query contains "Technical", filter only Technical tests
    if query.lower() == "technical":
        df = df[df["test_type"].str.lower() == "technical"]

    # ✅ If query contains "Cognitive", filter only Cognitive tests
    if query.lower() == "cognitive":
        df = df[df["test_type"].str.lower() == "cognitive"]

    # If no data matches the filter
    if df.empty:
        return []

    # Apply TF-IDF only on the filtered subset
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    query_vector = tfidf.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    return df.iloc[top_indices].to_dict(orient='records')



@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    recommendations = get_recommendations(query)
    return jsonify(recommendations)

if __name__ == '__main__':
    if not os.path.exists(DB_FILE):
        init_db()
        load_dummy_data()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

