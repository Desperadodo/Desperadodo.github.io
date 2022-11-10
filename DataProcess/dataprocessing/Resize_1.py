import cv2

img = cv2.imread(r'/Users/munros/Desktop/Sprint/98602905_p0.jpeg')

img = cv2.resize(img, (200, 200))

"""img = cv2.resize(img, (300, 300))"""

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
