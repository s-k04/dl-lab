import cv2
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("image.jpg", 0)

if img is None:
    print("Error: image.jpg not found")
    exit()

# Histogram Equalization
hist = cv2.equalizeHist(img)

# Thresholding
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Edge Detection
edge = cv2.Canny(img, 100, 200)

# Flipping
flip = cv2.flip(img, 1)

# Dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(img, kernel)

titles = ["Original", "Histogram", "Threshold",
          "Edge", "Flip", "Dilation"]

images = [img, hist, thresh, edge, flip, dilate]

plt.figure(figsize=(10, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()