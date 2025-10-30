#frontend/client.py
import os 
import requests


class LLMClient:
    def __init__(self, base_url=os.getenv("API_URL")):
        self.base_url = base_url
    def generate_text(self, prompt, temperature=0.2, max_tokens=1600):
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "query": prompt
        }
        try:
            
            response = requests.post(f"{self.base_url}/ask", json=payload, headers=headers)
            response.raise_for_status()
        except requests.HTTPError:
            return response.status_code, response.text
        return response.json()
    
    def send_feedback(self, payload: dict):
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(f"{self.base_url}/feedback", json=payload, headers=headers)
            response.raise_for_status()
            return True, response.json()
        except requests.RequestException as e:
            return False, {"error": str(e)}