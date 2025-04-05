# SHL Assessment Recommender 

A smart recommendation system that suggests the most relevant SHL individual test solutions based on a natural language job description or query.

---

##  Project Structure

```
shl-assessment-recommender/
├── app.py
├── database.py
├── model.py
├── recommend.py
├── data.json
├── shl_assessments.db
├── requirements.txt
└── README.md
```

---

##  Tech Stack

- Python
- Flask
- SQLite
- scikit-learn (TF-IDF & Cosine Similarity)

---


##  Features

- Accepts a job description or query as input
- Recommends up to 10 relevant SHL assessments
- Returns test name, URL, duration, type, and support info
- Uses TF-IDF + Cosine Similarity for semantic matching

---

# How It Works

1. A query is received at the `/recommend` POST endpoint.
2. TF-IDF vectorizer converts the query and test data into numerical vectors.
3. Cosine similarity calculates relevance scores.
4. Top results are returned as structured JSON.

---

## API Usage

**Endpoint:** `POST /recommend`

###  Request Body:
```json

{
  "query": "We are hiring a Python developer with good analytical skills"
}

```

---

###  Sample Response:
```json
[
  {
    "name": "Python & SQL Test",
    "url": "https://shl.com/python-sql",
    "duration": "60 mins",
    "test_type": "Technical",
    "adaptive_support": "No",
    "remote_support": "Yes"
  }
]
```

---


##  Running Locally

```bash
git clone https://github.com/your-username/shl-assessment-recommender.git
cd shl-assessment-recommender
pip install -r requirements.txt
python app.py
```

---

##  Hosted Demo

[Click here to try it live](https://shl-assessment-recommender-qhkg.onrender.com)

---

##  License

MIT License – feel free to use and enhance this project for learning and development.