#frontend/client.py
import os 
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, base_url=os.getenv("CHATENDPNT")):
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