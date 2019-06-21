# USAGE
# python video_facial.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from __future__ import division
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import math
import numpy as np

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
#print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
#print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		p = predictor(gray, rect)
		p = face_utils.shape_to_np(p)
		#print(points.shape)
		
		(x, y, w, h) = face_utils.rect_to_bb(rect)
	        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	        k = (p[19][1] - y)+(p[20][1] - y)+(p[21][1] - y)+(p[18][1] - y)
	        T = k/h
	        print "%.2f " % (T)
	        """
	        print("y0 :")
	        print(p[19][1])
	        print("y :")
	        print(y)
	        print("h :")
	        print(h) 
	        """
		# grt the normalizing area
		#Unr = (p[0][1]*p[16][0]) + (p[16][1]*p[30][0]) + (p[30][1]*p[0][0]) 
		#Lnr = (p[0][0]*p[16][1]) + (p[16][0]*p[30][1]) + (p[30][0]*p[0][1]) 
		#normalizing_area = (0.5)*abs(Unr - Lnr)
		#print(normalizing_area)
		
		#Um = (p[51][1]*p[48][0]) + (p[48][1]*p[57][0]) + (p[57][1]*p[54][0]) + (p[54][1]*p[51][0])
		#Lm = (p[51][0]*p[48][1]) + (p[48][0]*p[57][1]) + (p[57][0]*p[54][1]) + (p[54][0]*p[51][1])
		#mouth_area = (0.5)*abs(Um - Lm)
		#print(mouth_area/normalizing_area)
		
		#Ue = (p[21][1]*p[22][0]) + (p[22][1]*p[27][0]) + (p[27][1]*p[21][0]) 
		#Le = (p[21][0]*p[22][1]) + (p[22][0]*p[27][1]) + (p[27][0]*p[21][1]) 
		#eyebrows_area = (0.5)*abs(Ue - Le)
		#print(eyebrows_area)
		
		## Draw polyguns
		#pts = np.array([[p[21][0],p[21][1]],[p[22][0],p[22][1]],[p[27][0],p[27][1]]], np.int32)
    		#pts = pts.reshape((-1,1,2))
    		#cv2.polylines(frame,[pts],True,(0,255,255))
    		
    		#pts1 = np.array([[p[51][0],p[51][1]],[p[48][0],p[48][1]],[p[57][0],p[57][1]],[p[54][0],p[54][1]]], np.int32)
    		#pts1 = pts1.reshape((-1,1,2))
    		#cv2.polylines(frame,[pts1],True,(255,0,255))
    		
    		#cv2.line(frame,(p[39][0],p[39][1]),(p[21][0],p[21][1]),(255,0,0),2)
		#cv2.line(frame,(p[39][0],p[39][1]),(p[20][0],p[20][1]),(255,0,0),2)
		#cv2.line(frame,(p[39][0],p[39][1]),(p[19][0],p[19][1]),(255,0,0),2)
		#cv2.line(frame,(p[39][0],p[39][1]),(p[18][0],p[18][1]),(255,0,0),2)
		#cv2.line(frame,(p[39][0],p[39][1]),(p[17][0],p[17][1]),(255,0,0),2)
		
		"""
		dist_a = math.sqrt(math.pow((p[39][0] - p[21][0]),2)+math.pow((p[39][1] - p[21][1]),2))
		dist_b = math.sqrt(math.pow((p[39][0] - p[20][0]),2)+math.pow((p[39][1] - p[20][1]),2))
		dist_c = math.sqrt(math.pow((p[39][0] - p[19][0]),2)+math.pow((p[39][1] - p[19][1]),2))
		
		normlized_dist = (dist_a + dist_b +dist_c)/dist_a
		
		print(normlized_dist)
		"""
		#x = points[30][0]
		#y = points[30][1]
		#cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in p:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
