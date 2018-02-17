import cv2
import numpy as np


class TargetFinder ():
	def __init__ (self):
		self.boundaries = ((29, 110, 6), (80, 255, 220))


	def processShape (self, contour):
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
		return len(approx) == 4

	def processFrame (self, frame, boundaries):
		(lower, upper) = boundaries

		lower = np.array(lower, dtype="uint8")
		upper = np.array(upper, dtype="uint8")
		blurred = cv2.GaussianBlur(frame, (5, 5), 0)

		mask = cv2.inRange(blurred, lower, upper)
		kernel = np.ones((2,2),np.uint8)

		erosion = cv2.erode(mask, kernel, iterations=4)
		dilation = cv2.dilate(erosion, kernel, iterations=4)

		edged = cv2.Canny(dilation, 50, 150)

		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		# cnts = list(filter(lambda cnt: self.processShape(cnt), cnts))
		# center = None

		# loop over the contours

		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			
			try:
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			except:
				pass
	 
			# only proceed if the radius meets a minimum size
			if radius > 20:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
				
		return mask

# img = cv2.imread('photo.jpg')

# processed = finder.processFrame(img, boundaries=greenBoundaries)
# cv2.imshow('hue', processed)

# cv2.waitKey(0) 