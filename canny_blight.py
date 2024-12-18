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
    lower_yellowish_red = np.array([10, 50, 80])
    upper_yellowish_red = np.array([30, 255, 255])

    # "Gray chocolate-ish" range (adjust as needed)
    lower_chocolate = np.array([20, 20, 20])
    upper_chocolate = np.array([30, 255, 250])

    # Create masks for both colors
    mask_yellowish_red = cv2.inRange(hsv, lower_yellowish_red, upper_yellowish_red)
    mask_chocolate = cv2.inRange(hsv, lower_chocolate, upper_chocolate)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_yellowish_red, mask_chocolate)

    # Apply combined mask
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

    # Canny edge detection thresholds for masked image
    lower_thres = 220
    higher_thres = 250

    # Split masked image channels
    masked_r = masked_image[:, :, 2]
    masked_g = masked_image[:, :, 1]
    masked_b = masked_image[:, :, 0]

    # Canny on masked channels
    edges_r = cv2.Canny(masked_r, lower_thres, higher_thres)  
    edges_g = cv2.Canny(masked_g, 200, 255)                   
    edges_b = cv2.Canny(masked_b, 200, 255)                   

    edges_combined = cv2.merge([edges_b, edges_g, edges_r])
    edges_gray = cv2.cvtColor(edges_combined, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------
    # Original Canny (without masking) for comparison
    # -----------------------------------------
    # gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # original_edges = cv2.Canny(gray, 100, 250)  # Adjust thresholds if needed

    # Plot the results
    plt.figure(figsize=(20, 8))

    # Original Image
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Original Canny Edges
    # plt.subplot(1, 5, 2)
    # plt.title("Original Canny Edges")
    # plt.imshow(original_edges, cmap='gray')
    # plt.axis('off')

    # Masked Image
    plt.subplot(1, 4, 2)
    plt.title("Masked Image (Combined Colors)")
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Canny Edges (Separate RGB on Masked Image)
    plt.subplot(1, 4, 3)
    plt.title("Canny Edges on Mask")
    plt.plot([], [], color='r', label='Red Channel')
    plt.plot([], [], color='g', label='Green Channel')
    plt.plot([], [], color='b', label='Blue Channel')
    plt.imshow(cv2.cvtColor(edges_combined, cv2.COLOR_BGR2RGB))
    plt.legend()
    plt.axis('off')

    # Canny Edges (Combined Gray on Masked Image)
    plt.subplot(1, 4, 4)
    plt.title("Canny Edges (Combined Gray)")
    plt.imshow(edges_gray, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

compare_edge_detection(r"Dataset\Blight\Corn_Blight (1).jpeg")
# compare_edge_detection(r"C:\Users\Gung De\Documents\Kampus Stuff\Semester 3\PCD\opencv-edge-detection\Dataset\Healthy\Corn_Health (1).jpg")
# compare_edge_detection(r'C:\Users\Gung De\Documents\Kampus Stuff\Semester 3\PCD\Project-UAS-Kelompok-main\Plot Hasil Testing\Test 15 (Bil Filtering-5)\preprocessing_color_blight.png')
# compare_edge_detection(r"Dataset\Common_Rust\Corn_Common_Rust (250).jpg")
