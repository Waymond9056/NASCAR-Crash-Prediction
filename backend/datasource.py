import requests
import json
url = 'https://cf.nascar.com/cacher/2022/1/5150/lap-times.json'

response = requests.get(url)

with open("Test.json", "w") as file:
    json.dump(json.loads(response.text),file)
