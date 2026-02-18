import requests

class EMRClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def fetch_record(self, patient_id):
        r = requests.get(f"{self.base_url}/records/{patient_id}", headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()