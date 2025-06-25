import re
import pandas as pd
import matplotlib.pyplot as plt

# Load and parse the text file
with open("evaluation_results_2.txt", "r") as file:
    raw_text = file.read()

# Regular expression to extract all the required data
pattern = re.compile(
    r'Model (\w+):\s+Success rate:\s+([\d.]+)%.*?Avg a_error:\s+([\d.]+) km\s+Avg steps:\s+([\d.]+)\s+Avg fuel used:\s+([\d.]+) kg',
    re.DOTALL
)
matches = pattern.findall(raw_text)

# Convert to DataFrame
df = pd.DataFrame(matches, columns=['Model', 'SuccessRate', 'AError', 'Steps', 'FuelUsed'])
df['SuccessRate'] = df['SuccessRate'].astype(float)
df['AError'] = df['AError'].astype(float)
df['Steps'] = df['Steps'].astype(float)
df['FuelUsed'] = df['FuelUsed'].astype(float)

# Extract penalty values from model ID
df['t'] = df['Model'].apply(lambda x: int(x.split('_')[0][:2]))
df['m'] = df['Model'].apply(lambda x: int(x.split('_')[1][:-1]))

# Create scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Steps'], df['FuelUsed'], c=df['SuccessRate'], cmap='viridis', s=100, edgecolors='k')

# Annotate points with model names
for _, row in df.iterrows():
    plt.text(row['Steps'] + 100, row['FuelUsed'] + 0.2, row['Model'], fontsize=9)

# Add colorbar and labels
cbar = plt.colorbar(scatter)
cbar.set_label('Success Rate (%)')

plt.xlabel('Average Steps')
plt.ylabel('Average Fuel Used (kg)')
plt.title('Fuel vs. Steps Across Reward Shaping Configurations')
plt.grid(True)
plt.tight_layout()
plt.show()
