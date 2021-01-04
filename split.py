import json 
import random 

data_path = 'data/data.json' 
reference_path = 'data/LCCC-base_train.json'

with open(data_path, 'r', encoding='utf-8') as f: 
    data = json.load(f) 

print(len(data.keys()))

with open(reference_path, 'r', encoding='utf-8') as f:
    reference = json.load(f)


reference_list = []
for reference_sentence in reference: 
    tmp_reference = reference_sentence[0].replace(' ', '').strip()
    #print(tmp_reference)
    reference_list.append(tmp_reference) 



# train: valid: test = 8:1:1
train_id = []
valid_id = []
test_id = []

for dialog_id in data.keys():
    try: 
        if data[dialog_id][0]['txt'] in reference_list:
            train_id.append(dialog_id)
    except:
        print(data[dialog_id])

print(len(train_id))

# the residual will be split randomly 

 

