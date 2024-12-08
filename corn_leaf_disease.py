import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

dataset_dir = r'D:\1_Kuliah Gusmang\Semester_3\Kuliah\1_Pengolahan_Citra_Digital\Project-UAS-Kelompok\Dataset'

images = []
labels = []

label_map = {'Healthy': 0, 'Blight': 1, 'Common_Rust': 2, 'Gray_Leaf_Spot': 3,}

def load_data(dataset_dir, label_map):
    images = []
    labels = []
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

    return images, labels

images, labels = load_data(dataset_dir, label_map)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, (label, label_idx) in enumerate(label_map.items()):
    
    idxs = np.where(labels == label_idx)[0]
    
    img = images[idxs[0]]
    
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('Plot Hasil Testing/baca_awal.png', bbox_inches='tight')
# plt.show()

def preprocess_images(images):
    processed_images = []

    for img in images:
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Konversi dari RGB ke YCrCb
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(img_ycrcb)

        # Menggunakan Gaussian Blur pada saluran Y
        # y_blurred = cv2.GaussianBlur(y, (15, 15), 0)
        
        # Equalisasi histogram pada saluran Y
        y_eq = cv2.equalizeHist(y)
        img_ycrcb_eq = cv2.merge([y_eq, cr, cb])

        # img_blur = cv2.GaussianBlur(img_ycrcb_eq, (3, 3), 0)

        # Kembali ke RGB
        img_rgb = cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2RGB)

        # Normalisasi
        img_normalized = img_rgb / 255.0
        processed_images.append(img_normalized)

    return np.array(processed_images)

images, labels = load_data(dataset_dir, label_map)

processed_images = preprocess_images(images)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, (label, label_idx) in enumerate(label_map.items()):
    
    idxs = np.where(labels == label_idx)[0]
    
    img = processed_images[idxs[0]]
    
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('Plot Hasil Testing/preprocessing.png', bbox_inches='tight')
# plt.show()
