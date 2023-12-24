import matplotlib.pyplot as plt
import json

with open('news_data.json') as file:
    obs = json.load(file)

data = []
for item in obs:
    data.append(item['view'])

# Create a figure and a plot
plt.figure(figsize=(10, 2))  # Set the figure size

# Plot each data point as a distinct marker on the number line
plt.scatter(data, [1]*len(data), color='blue', s=50)  # 's' is the size of the marker

# Set the limits and labels
plt.xlim(min(data) - 1, max(data) + 1)  # Set x-axis limits
plt.ylim(0, 2)  # Set y-axis limits (not really important here, just for visual)
plt.yticks([])  # Hide y-axis ticks
plt.title('Number Line with Data Points')

# Show the plot
plt.savefig('distribution.png', bbox_inches='tight')
