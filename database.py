import json

def load_assessments():
    with open("data.json", "r") as f:
        return json.load(f)