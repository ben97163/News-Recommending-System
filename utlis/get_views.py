from datetime import datetime
import numpy as np
import json
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

target_date = -1
for key, value in views_by_date.items():
    if key > target_date:
        target_date = key

target_data = np.log(np.array([ data[0] for data in views_by_date[target_date]]))

mean_value = np.mean(target_data)

for key, values in views_by_date.items():
    if key == target_date:
        for i in range(len(values)):
            output_dict.append({'title': values[i][1], 'view': target_data[i], 'content': values[i][2]})
    else:
        array = np.array([ data[0] for data in views_by_date[key]])
        safe_array = np.clip(array, 1e-10, None)

        log_data = np.log(safe_array)
        log_data = np.clip(log_data, 1e-10, None)
        mean_log = np.mean(log_data)
        log_data = log_data / (mean_log / mean_value)

        for i in range(len(values)):
            output_dict.append({'title': values[i][1], 'view': log_data[i], 'content': values[i][2]})

with open('total_views.json', 'w', encoding='utf-8') as file:
    json.dump(output_dict, file, ensure_ascii=False, indent=4)




    