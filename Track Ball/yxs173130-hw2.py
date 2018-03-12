import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
import argparse
import imutils
import pylab as pl

class mobileBall:

	def __init__(self):
		self.numFrame = 0

	def cal_real_value(self, box, y_ori, x_ori, length, height):
		xlist=[box[0][0], box[1][0], box[2][0], box[3][0]]
		xlist.sort()
		ylist=[box[0][1], box[1][1], box[2][1], box[3][1]]
		ylist.sort()
		y1=int((ylist[1] + ylist[0])/2)
		y2=int((ylist[2] + ylist[3])/2)

		y_real = ((y_ori-y1)/(y2-y1)) * height
		x_l = xlist[1] - ((xlist[1]-xlist[0])/(y2-y1))*(y_ori-y1)
		x_r = xlist[2]+ ((xlist[3]-xlist[1])/(y2-y1))*(y_ori-y1)
		x_real = ((x_ori - x_l)/(x_r)) * length
		return x_real, y_real

	def find_averAreaRect(self, img, box):
	    xlist=[box[0][0],box[1][0],box[2][0],box[3][0]]
	    xlist.sort()
	    ylist=[box[0][1],box[1][1],box[2][1],box[3][1]]
	    ylist.sort()

	    x1 = int((xlist[1] + xlist[0])/2)
	    y1 = int((ylist[1] + ylist[0])/2)
	    height = int((abs((ylist[3]-ylist[0])) + abs((ylist[2]-ylist[1])))/2) - 8
	    width = int((abs((xlist[3]-xlist[0])) + abs((xlist[2]-xlist[1])))/2)
	    averRect = img[y1: y1 + height, x1: x1 + width]

	    return averRect, x1, y1

	def trackBall(self):
		flag = None
		oldCenter = None
		center = None
		capture = cv2.VideoCapture('cs6327-a2-demo-edit.mp4')
		fps = capture.get(cv2.CAP_PROP_FPS)
		img_width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		img_height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		ret, old_frame = capture.read()
		showline = np.zeros_like(old_frame)
		blackbg = np.zeros((img_height, img_width, 3), np.uint8)
		speedlist = []
		real_speedlist = []
		greenLower = (48, 87, 65)
		greenUpper = (77, 200, 120)
		whiteLower = (43, 37, 140)
		whiteUpper = (58, 104, 222)

		while ret:
			ret, frame = capture.read()
			self.numFrame = self.numFrame + 1
			if not ret:
				break
			else:
				frame = cv2.GaussianBlur(frame, (5, 5), 0)
				hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				mask = cv2.inRange(hsv, greenLower, greenUpper)
				mask = cv2.erode(mask, None, iterations=2)
				mask = cv2.dilate(mask, None, iterations=2)

				tableContour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
				c1 = max(tableContour, key = cv2.contourArea)
				rect = cv2.minAreaRect(c1)
				box = cv2.boxPoints(rect)
				box = np.int0(box)

				averRect, x0, y0= self.find_averAreaRect(frame, box)
				hsv2 = cv2.cvtColor(averRect, cv2.COLOR_BGR2HSV)
				mask2 = cv2.inRange(hsv2, whiteLower, whiteUpper)
				mask2 = cv2.erode(mask2, None, iterations=2)
				mask2 = cv2.dilate(mask2, None, iterations=1)
				ballContour = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

				if len(ballContour) > 0:
					maxBallArea = max(ballContour, key=cv2.contourArea)
					((x, y), radius) = cv2.minEnclosingCircle(maxBallArea)
					moment = cv2.moments(maxBallArea)
					center = (int(moment["m10"] / moment["m00"]) + x0, int(moment["m01"] / moment["m00"]) + y0)

				if flag is None:
					oldCenter = center
					flag = False

				if radius > 1:
					cv2.circle(frame, (int(x+x0), int(y+y0)), int(radius),(0, 255, 0), 2)
					showline = cv2.line(showline, oldCenter, center, (255, 255, 255), 5)

					xi = center[0]
					yi = center[1]
					x0 = oldCenter[0]
					y0 = oldCenter[1]
					x_real, y_real = self.cal_real_value(box, yi, xi, 177.8, 356.9)
					x_real_0, y_real_0 = self.cal_real_value(box,y0,x0, 177.8, 356.9)
					real_distance = np.sqrt(np.square(x_real - x_real_0) + np.square(y_real - y_real_0))
					real_speed = real_distance * fps
					real_speedlist.append(real_speed)
					distance = np.sqrt(np.square(xi - x0) + np.square(yi - y0))
					speed = distance * fps
					speedlist.append(speed)
					oldCenter = center

				frame = imutils.resize(frame, width=800)
				cv2.imshow('Ball Detection',frame)
				blackbg = cv2.add(blackbg, showline)
				blackbg_sm = cv2.resize(blackbg,(800, 450))
				cv2.imshow('Paths', blackbg_sm)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		self.drawHistogram(speedlist, real_speedlist, fps)
		capture.release()

	def drawHistogram(self,speedlist, real_speedlist, fps):
		y = speedlist
		y2 = real_speedlist
		x = [(x/fps) for x in range(1, self.numFrame)]
		pl.plot(x, y)
		pl.plot(x, y2)
		pl.title('Speed of the cue ball ')
		pl.xlabel('Time (second)')
		pl.ylabel('Blue - Speed(pixel/s)  Orangle - Speed(cm/s)')
		pl.show()

	def main(self):
		self.trackBall()

if __name__ == "__main__":
	mobileBall().main()
