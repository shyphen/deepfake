import requests

url = "http://127.0.0.1:5000/predict"
files = {"image": open("me.jpg", "rb")}  # Replace with actual image path

response = requests.post(url, files=files)
print(response.json())
