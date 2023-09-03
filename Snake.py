import math
import random

import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
success, img = cap.read()
cap.set(3, 1366)
cap.set(4, 768)

detector = HandDetector(detectionCon=0.8, maxHands=1)


def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull[pos[1]:pos[1] + hf, pos[0]:pos[0] + wf, :] = imgRGB

    imgBack = cv2.bitwise_and(imgBack, cv2.bitwise_not(maskBGR))
    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack


class NokiaSnakeGame:
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distances between each point
        self.currentLength = 0  # Total length of the snake
        self.allowedLength = 150  # Total allowable length
        self.previousHead = (0, 0)  # previous head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.imgFood = cv2.resize(self.imgFood, (512, 512))

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoints = (0, 0)
        self.randomFood()
        self.score = 0

    def randomFood(self):
        self.foodPoints = (
            random.randint(100, 1000),
            random.randint(100, 600)
        )

    def update(self, imgMain, currentHead):
        px, py = self.previousHead
        cx, cy = currentHead

        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = (cx, cy)

        # Length Reduction
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength < self.allowedLength:
                    break

        # Check for food eaten
        rx, ry = self.foodPoints
        if (
                rx - self.wFood // 2 < cx < rx + self.wFood // 2 and
                ry - self.hFood // 2 < cy < ry + self.hFood // 2
        ):
            self.randomFood()
            self.allowedLength += 50
            self.score += 1
            print(self.score)

            # Draw snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                cv2.circle(imgMain, self.points[-1], 10, (200, 0, 200), cv2.FILLED)

            # Draw Food
            rx, ry = self.foodPoints
            imgMain = overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

            return imgMain

# Creating The Actual Game Here
game = NokiaSnakeGame("chicken.png")

# Create a named window and set it to full screen
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

