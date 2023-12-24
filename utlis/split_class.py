import json

with open('total.json') as file:
    data = json.load(file)

label0 = []
label1 = []
label2 = []

for item in data:
    if item["label"] == 0:
        label0.append(item)
    elif item["label"] == 1:
        label1.append(item)
    else:
        label2.append(item)

split = int(len(label0) / 10)
val = []
train = []
for i in range(split):
    val.append(label0[i])
    val.append(label1[i])
    val.append(label2[i])

for i in range(split, len(label0)):
    train.append(label0[i])
for i in range(split, len(label1)):
    train.append(label1[i])
for i in range(split, len(label2)):
    train.append(label2[i])

print(len(val))
print(len(train))

with open('val.json', 'w', encoding='utf-8') as file:
    json.dump(val, file, ensure_ascii=False, indent=4)
with open('train.json', 'w', encoding='utf-8') as file:
    json.dump(train, file, ensure_ascii=False, indent=4)
