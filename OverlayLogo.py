import cvzone
import cv2

imgBack = cv2.imread("Resources/pc.jpg")
imgFront = cv2.imread("Resources/logo.png", cv2.IMREAD_UNCHANGED)
imgFront = cv2.resize(imgFront, (0, 0), None, 0.3, 0.3)

hf, wf, cf = imgFront.shape
hb, wb, cb = imgBack.shape

imgResult = cvzone.overlayPNG(imgBack, imgFront, [0, hb-hf])

cv2.imshow("Image", imgResult)
cv2.waitKey(0)
