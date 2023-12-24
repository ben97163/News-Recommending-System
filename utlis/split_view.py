import json

title_map = {}
with open('total_views.json') as file:
    data = json.load(file)

for idx, instance in enumerate(data):
    title_map[instance["title"]] = idx

with open('train.json') as file:
    train_data = json.load(file)

for idx, instance in enumerate(train_data):
    original_idx = title_map[instance["title"]]
    train_data[idx]['view'] = data[original_idx]['view']

with open('val.json') as file:
    val_data = json.load(file)

for idx, instance in enumerate(val_data):
    original_idx = title_map[instance["title"]]
    val_data[idx]['view'] = data[original_idx]['view']

with open('train_view.json', 'w', encoding='utf-8') as file:
    json.dump(train_data, file, ensure_ascii=False, indent=4)

with open('val_view.json', 'w', encoding='utf-8') as file:
    json.dump(val_data, file, ensure_ascii=False, indent=4)

