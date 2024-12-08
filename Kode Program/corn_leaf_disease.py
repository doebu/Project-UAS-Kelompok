import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

dataset_dir = r'D:\1_Kuliah Gusmang\Semester_3\Kuliah\1_Pengolahan_Citra_Digital\Project UAS Kelompok\Dataset'

images = []
labels = []

label_map = {'Healthy': 0, 'Blight': 1, 'Common_Rust': 2, 'Gray_Leaf_Spot': 3,}

for label, label_idx in label_map.items():
    label_dir = os.path.join(dataset_dir, label)

    for filename in os.listdir(label_dir):
        img_path = os.path.join(label_dir, filename)

        img = cv2.imread(img_path)

        img = cv2.resize(img, (128, 128))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img_rgb)
        labels.append(label_idx)


images = np.array(images)
labels = np.array(labels)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, (label, label_idx) in enumerate(label_map.items()):
    
    idxs = np.where(labels == label_idx)[0]
    
    img = images[idxs[0]]
    
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.tight_layout()
plt.show()