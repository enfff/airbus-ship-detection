import matplotlib.pyplot as plt
import numpy as np

# Data
species = [
    'Original',
    'Gaussian Noise',
    'Gaussian Blur',
    'Motion Blur',
    'Brightness',
    'Fog'
]

# Normalized data (divided by 0.5)
penguin_means = {
    'No Augmentation': (0.4630 / 0.5, 0.2518 / 0.5, 0.2273 / 0.5, 0.0837 / 0.5, 0.3500 / 0.5, 0.1152 / 0.5),
    'Geometric': (0.4280 / 0.5, 0.2250 / 0.5, 0.2007 / 0.5, 0.0745 / 0.5, 0.2814 / 0.5, 0.0749 / 0.5),
    'Auxiliary Fourier Basis': (0.4550 / 0.5, 0.2956 / 0.5, 0.2573 / 0.5, 0.0772 / 0.5, 0.2729 / 0.5, 0.0874 / 0.5),
    'Patch Gaussian': (0.4733 / 0.5,
                       0.2788 / 0.5,
                       0.2441 / 0.5,
                       0.0791 / 0.5,
                       0.3805 / 0.5,
                       0.0784 / 0.5,),
}


x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars

# Set figure size
fig, ax = plt.subplots(figsize=(12, 8))

# Plotting each group of bars
for i, (label, values) in enumerate(penguin_means.items()):
    ax.bar(x + i * width, values, width, label=label)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized mAP')
ax.set_title('Normalized Augmentation Effects on Different Conditions')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(species)
ax.legend(title='Augmentation Type')
ax.set_ylim(0, 1.0)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
