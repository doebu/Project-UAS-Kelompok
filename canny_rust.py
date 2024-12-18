import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_edge_detection(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: File does not exist - {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image: {image_path}")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Original "yellowish-red" range
    lower_yellowish_red = np.array([0, 100, 0]) 
    upper_yellowish_red = np.array([30, 255, 255])

    # "Black-ish" range (tweak as needed)
    # Since black depends mostly on low value, we keep a wide hue & saturation range
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255,40])  # V<=50 considered dark/blackish

    # Create masks
    mask_yellowish_red = cv2.inRange(hsv, lower_yellowish_red, upper_yellowish_red)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Combine masks using bitwise_or to detect either color
    combined_mask = cv2.bitwise_or(mask_yellowish_red, mask_black)

    # Apply combined mask
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

    # Canny edge detection thresholds for masked image
    lower_thres = 50
    higher_thres = 180

    # Split masked image channels
    masked_r = masked_image[:, :, 2]
    masked_g = masked_image[:, :, 1]
    masked_b = masked_image[:, :, 0]

    # Canny on masked channels
    edges_r = cv2.Canny(masked_r, lower_thres, higher_thres)  
    edges_g = cv2.Canny(masked_g, 80, 255)                   
    edges_b = cv2.Canny(masked_b, 80, 255)                   

    edges_combined = cv2.merge([edges_b, edges_g, edges_r])
    edges_gray = cv2.cvtColor(edges_combined, cv2.COLOR_BGR2GRAY)

    # Convert original to RGB for display
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create an overlay by copying original
    overlay = original_rgb.copy()

    # Color the edges red (no blending, direct assignment)
    overlay[edges_gray > 0] = [255, 0, 0]

    # Plot the results
    plt.figure(figsize=(20, 8))

    # Original Image
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    # Masked Image
    plt.subplot(1, 4, 2)
    plt.title("Masked Image (Yellowish-Red + Black)")
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Canny Edges (Combined Gray)
    plt.subplot(1, 4, 3)
    plt.title("Canny Edges (Combined Gray)")
    plt.imshow(edges_gray, cmap='gray')
    plt.axis('off')

    # Canny Edges Overlay on Original
    plt.subplot(1, 4, 4)
    plt.title("Canny Edges Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
compare_edge_detection(r"Dataset\Common_Rust\Corn_Common_Rust (1020).jpg")
