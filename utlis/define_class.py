from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
# Provided JSON data
with open('news_data.json') as file:
    data = json.load(file)

output_dict = []
# Parse the dates and views
views_by_date = {}
total_views = []
for item in data:
    date = datetime.strptime(item['time']['date'], "%Y-%m-%d %H:%M")
    week = (date.isocalendar()[1] - 1) // 2# converting week to 2-week period
    if week not in views_by_date:
        views_by_date[week] = []
    views_by_date[week].append((item['view'], item['title'], item['content'], item['time']['date']))

# Calculate mean and variance for each 2-week period
for week, values in views_by_date.items():
    values.sort(key=lambda x: x[0], reverse=True)
    bounding1 = int(len(values) / 3)
    bounding2 = int(len(values) * 2 / 3)

    for i in range(len(values)):
        if i <= bounding1:
            output_dict.append({'title': values[i][1], 'label': 0, 'content': values[i][2]})
        elif i > bounding1 and i <= bounding2:
            output_dict.append({'title': values[i][1], 'label': 1, 'content': values[i][2]})
        else:
            output_dict.append({'title': values[i][1], 'label': 2, 'content': values[i][2]})

with open('total.json', 'w', encoding='utf-8') as file:
    json.dump(output_dict, file, ensure_ascii=False, indent=4)
