import math
import random

import cvzone
import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
success, img = cap.read()
cap.set(3, 1366)
cap.set(4, 768)

detector = HandDetector(detectionCon=0.8, maxHands=2)

class NokiaSnakeGame:
    def __init__ (self,pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distances between each point
        self.currentLength = 0  # Total length of the snake
        self.allowedLength = 150  # Total allowable length
        self.previousHead = 0, 0  # previous head point
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_COLOR)
        self.imgFood = cv2.resize(self.imgFood, (75, 75))
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.imgFood = cv2.resize(self.imgFood, (75, 75))
        self.imgFood = cv2.cvtColor(self.imgFood, cv2.COLOR_BGRA2BGR)


        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoints = 0,0
        self.randomFood()
        self.score = 0

    def randomFood(self):
        self.foodPoints = random.randint(100, 1000), random.randint(100,600)
    def update(self, imgMain, currentHead):
        px, py = self.previousHead
        cx, cy = currentHead  # Fixed typo here

        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        #Length Redduction
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength<self.allowedLength:
                    break

        #Cheak for food eaten
        rx, ry = self.foodPoints
        if rx - self.wFood//2 <cx < rx + self.wFood//2 and \
            ry - self.hFood//2 < cy < ry + self.hFood//2:
            self.randomFood()
            self.allowedLength += 50
            self.score += 1
            print(self.score)

        #Draw snake
        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
            cv2.circle(imgMain, self.points[-1], 10, (200, 0, 200), cv2.FILLED)  # Fixed variable name here
            return imgMain

        #Draw Food
        rx, ry = self.foodPoints
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood//2, ry - self.hFood//2))
        return imgMain

# Creating The Actual Game Here
game = NokiaSnakeGame("chicken.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
