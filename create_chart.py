import matplotlib.pyplot as plt
import seaborn as sns

# Configure Seaborn style for a vivid look
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)

# Ensure Times New Roman is used globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Data
methods = ['AlexNet + RF\nKursun and Koklu', 'VGG16\n(this study)', 'VGG16 + ET\n(this study)']
accuracies = [74.64, 67.97, 81.66]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Use Seaborn barplot with a vivid palette
# Setting hue to methods and legend=False is recommended in recent Seaborn versions
sns.barplot(x=methods, y=accuracies, palette="deep", hue=methods, legend=False, ax=ax)

# Limit y-axis from 50 to 95 for better visual comparison
ax.set_ylim(50, 95)
ax.set_ylabel('Accuracy (%)', fontname='Times New Roman', fontsize=14)
ax.set_xlabel('')

# Add data labels on top of the bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.2f}%',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontname='Times New Roman', fontsize=14)

# Remove top and right spines to keep it clean, even with grid
sns.despine()

# Adjust layout
plt.tight_layout()

# Save as PDF
output_path = r'e:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\accuracy_comparison_chart.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight')

print(f"Chart successfully saved to {output_path}")
