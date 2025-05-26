import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

def load_and_resize(image_path, size=(512, 512)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image = cv2.resize(image, size)
    return image

def align_images(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB detector to find keypoints and descriptors
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Find homography matrix
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp second image to align with the first
    height, width = img1.shape[:2]
    aligned_img2 = cv2.warpPerspective(img2, H, (width, height))
    return aligned_img2

def detect_changes_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 230, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the regions that changed
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img2.copy()
    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return result, diff, thresh

def visualize_results(img1, img2, diff, thresh, result):
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Image T1")
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Image T2 (Aligned)")
    axs[2].imshow(diff, cmap='gray')
    axs[2].set_title("SSIM Difference")
    axs[3].imshow(thresh, cmap='gray')
    axs[3].set_title("Thresholded")
    axs[4].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axs[4].set_title("Detected Changes")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def run_change_detection(image1_path, image2_path):
    print("Loading images...")
    img1 = load_and_resize(image1_path)
    img2 = load_and_resize(image2_path)

    print("Aligning images...")
    aligned_img2 = align_images(img1, img2)

    print("Detecting changes...")
    result, diff, thresh = detect_changes_ssim(img1, aligned_img2)

    print("Visualizing results...")
    visualize_results(img1, aligned_img2, diff, thresh, result)

# ----------- Example Usage -----------
if __name__ == "__main__":
    # Replace these paths with your own satellite images
    image1_path = "satellite_t1.jpg"
    image2_path = "satellite_t2.jpg"

    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print(f"Make sure '{image1_path}' and '{image2_path}' exist in the current folder.")
    else:
        run_change_detection(image1_path, image2_path)
