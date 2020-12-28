import json 

data_path = 'data/data.json' 

with open(data_path, 'r', encoding='utf-8') as f: 
    data = json.load(f) 

print(len(data.keys()))