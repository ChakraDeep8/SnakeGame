import cvzone
import cv2
import numpy as np

angle = 0
fpsReader = cvzone.FPS()


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 100)
cv2.createTrackbar("Speed", "Parameters", 1, 25, empty)

while True:
    imgBack = np.ones((500, 800, 3), np.uint8) * 255
    imgG1 = cv2.imread("Resources/gear.png", cv2.IMREAD_UNCHANGED)
    imgG2 = imgG1.copy()

    val = cv2.getTrackbarPos("Speed", "Parameters")
    imgG1 = cvzone.rotateImage(imgG1, angle + 23)
    imgG2 = cvzone.rotateImage(imgG2, -angle)
    angle += val

    imgResult = cvzone.overlayPNG(imgBack, imgG1, [125, 100])
    imgResult = cvzone.overlayPNG(imgResult, imgG2, [400, 100])
    _, imgResult = fpsReader.update(imgResult)

    cv2.imshow("Image", imgResult)
    cv2.waitKey(1)
