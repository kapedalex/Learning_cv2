import cv2
import matplotlib.pyplot as plt

img = cv2.imread('src/haski.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('src/koshka.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, dsize=img.shape[::-1])
result_and = cv2.bitwise_and(img, img2, mask=None)
result_or = cv2.bitwise_or(img, img2, mask=None)
result_xor = cv2.bitwise_xor(img, img2, mask=None)
result_not = cv2.bitwise_not(img, mask=None)

plt.figure(figsize=[18, 5])
plt.subplot(221); plt.imshow(result_and, cmap='gray');
plt.subplot(222); plt.imshow(result_or, cmap='gray');
plt.subplot(223); plt.imshow(result_xor, cmap='gray');
plt.subplot(224); plt.imshow(result_not, cmap='gray');
plt.show()
