import json
import requests


API_URL = "http://127.0.0.1:8000/predict"

with open("api_test_request.json", "r", encoding="utf-8") as file:
    payload = json.load(file)

response = requests.post(API_URL, json=payload, timeout=30)
print("Status code:", response.status_code)
print(response.json())
