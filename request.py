import requests

url = 'http://localhost:5001/predict_api'
r = requests.post(url,json={'picklocation':(4,2)})

print(r.json())