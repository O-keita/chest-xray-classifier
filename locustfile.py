from locust import HttpUser, task, between

class ChestXrayUser(HttpUser):
    wait_time = between(1, 2.5)  # Simulate think time between requests

    @task
    def get_status(self):
        self.client.get("/status")

    @task
    def predict_dummy(self):
        with open("/home/omar/Downloads/archive (25)/chest_xray/test/NORMAL/IM-0001-0001.jpeg", "rb") as f:
            files = {'file': ("IM-0001-0001.jpeg", f, "image/jpeg")}
            self.client.post("/predict", files=files)