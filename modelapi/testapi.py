import requests

url = "http://localhost:9550/embed"
data = {
    "texts": ["这是第一句话", "这是第二句话"]
}
response = requests.post(url, json=data)
print(response.json())
