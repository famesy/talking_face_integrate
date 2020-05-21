import cv2

i = 12

img = cv2.imread('./demo_images/test_sample' + str(i) + '.jpg', cv2.IMREAD_UNCHANGED)
output = cv2.resize(img, (256,256), interpolation= cv2.INTER_AREA)
cv2.imwrite('./demo_images/test_sample' + str(i) + '.jpg',output)
