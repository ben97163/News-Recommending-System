import matplotlib.pyplot as plt
import json
with open('ppl.json','r') as file:
    ppls = json.load(file)

# Steps (assuming they are 100 steps apart)
steps = list(range(0, len(ppls) * 100, 100))

# Plotting the figure
plt.figure(figsize=(10, 6))
plt.plot(steps, ppls, marker='o')
plt.title('ppl on public test set per 100 Steps')
plt.xlabel('Steps')
plt.ylabel('ppl')
plt.grid(True)

# Save the figure as a PNG file
plt.savefig('pulic_ppl_plot.png')

