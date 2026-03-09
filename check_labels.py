import cv2
img = cv2.imread('download.jpg')
if img is not None:
    print(img.shape)
else:
    print("Could not load download.jpg")
