from imutils import face_utils, resize
import numpy as np
import dlib
import cv2
import csv
from shapely.geometry import Point,Polygon

variants = []

def numpixels(p1,clone):
	minx, miny, maxx, maxy = p1.bounds
	minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
	box_patch = [[x,y] for x in range(minx,maxx+1) for y in range(miny,maxy+1)]
	pixelsNum = 0
	totalAlpha = totalBeta = 0
	for pb in box_patch:
		pt = Point(pb[0],pb[1])
		if(p1.contains(pt)):
			pixelsNum += 1
			totalAlpha += clone[pb[0],pb[1]][1]
			totalBeta += clone[pb[0],pb[1]][2]
			clone[pb[0],pb[1]] = (255,255,255)
		else:
			clone[pb[0],pb[1]] = (0,0,0)

	# cv2.imshow("check",clone)
	# cv2.waitKey(0)
	return (pixelsNum,totalAlpha,totalBeta)

def find_pixels(i,j,shape,clone):	
	# size = j-i+1
	coords = []
	# x and y are interchanged
	for (y, x) in shape[i:j]:
		coords.append((x,y))
	p1 = Polygon(coords)
	return numpixels(p1,clone)


def findPixel(i,j,shape,clone,midNose,midJaw):
	coords = []
	for (y,x) in shape[i:j]:
		coords.append((x,y))
	p1 = Polygon(coords)
	minx, miny, maxx, maxy = p1.bounds
	minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
	box_patch = [[x,y] for x in range(minx,maxx+1) for y in range(miny,maxy+1)]
	leftNum = rightNum = 0
	# Slope of line
	slope = (midNose[1]-midJaw[1])/(midNose[0]-midJaw[0])
	for pb in box_patch:
		pt = Point(pb[0],pb[1])
		if(p1.contains(pt)):
			#make a line and use it see what is right and what is left
			val = (pb[1]-midJaw[1]) - slope*(pb[0]-midJaw[0])						
			if val > 0:
				leftNum += 1
				clone[pb[0],pb[1]] = (0,255,0)
			elif val == 0:
				clone[pb[0],pb[1]] = (255,255,255)
			else:
				rightNum += 1
				clone[pb[0],pb[1]] = (255,0,0)
		else:
			clone[pb[0],pb[1]] = (0,0,0)

	# cv2.imshow("check",clone)
	# cv2.waitKey(0)
	return (leftNum, rightNum)

	



# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
imagePath = 'images/1.jpeg'
image = cv2.imread(imagePath)
image = resize(image,width=200)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

var1 = var2 = var3 = var4 = var5 = var6 = 0
# detect faces in the grayscale image
rects = detector(gray, 1)


#for all variants except var6 in looping

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# print(face_utils.FACIAL_LANDMARKS_IDXS.items())
	clone = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	#Required
		# alpha(eyes)= alpha(leftEye)+alpha(rightEye), numPixels(eyes)= numPixels(leftEye)+numPixels(rightEye)
		# beta(eyes)= beta(leftEye)+beta(rightEye) 
		# alpha(lips)= alpha(mouth), numPixels(lips)=numPixels(mouth)

	# Given
		# ([('mouth', (48, 68)), ('inner_mouth', (60, 68)), ('right_eyebrow', (17, 22)), 
		# ('left_eyebrow', (22, 27)), ('left_eye', (36, 42)), ('right_eye', (42, 48)),
		# ('nose', (27, 36)), ('jaw', (0, 17))])

	# test purposes (individual parts)
	# copy1 = image.copy()	
	# for (x, y) in shape[i:j]:
	# 	copy1[y,x] = (255,255,255)
	# 	# cv2.circle(copy1, (x, y), 1, (0, 0, 255), -1)
	# cv2.imshow('image',copy1)
	# cv2.waitKey(0)

	# for Left eye
	(i,j) = (36,42)
	numLeye, alphaLeye, betaLeye = find_pixels(i,j,shape,clone)

	# for Right eye
	(i,j) = (42,48)
	numReye, alphaReye, betaReye = find_pixels(i,j,shape,clone)

	numEye = numLeye+numReye
	alphaEye = alphaLeye+alphaReye
	betaEye = betaLeye+betaReye

	var1 = alphaEye/numEye
	var2 = betaEye/numEye
	var5 = numLeye/numReye

	#for mouth
	(i,j) = (48,68)
	numMouth, alphaMouth, betaMouth = find_pixels(i,j,shape,clone)

	var3 = alphaMouth/numMouth

	#for var4 we need the middle contour of the face
	# For Variant 4, the middle line of a face was defined as the line passing through the middle label of the nose and the middle label of the face contour because the labels of ASM algorithm were indexed.
	
	#Finding the middle label element in nose 
	(i,j) = (27, 36)
	nose_list = []
	for (y,x) in shape[i:j]:
		clone[x,y] = (0,0,255) 
		nose_list.append((x,y))
	nose_list.sort(key = lambda x : x[1])
	midNose = nose_list[int(len(nose_list)/2)]

	#Finding the middle label element in face contour
	(i,j) = (0,17)
	jaw_list = []
	for (y,x) in shape[i:j]:
		clone[x,y] = (0,0,255) 
		jaw_list.append((x,y))
	jaw_list.sort(key = lambda x : x[1])
	midJaw = jaw_list[int(len(jaw_list)/2)]
	
	# test purposes
	# cv2.circle(clone,(midNose[1],midNose[0]),1,(255,255,0),-1)
	# cv2.circle(clone,(midJaw[1],midJaw[0]),1,(255,0,0),-1)
	# cv2.imshow("check",clone)
	# cv2.waitKey(0)


	# for variant 4
	leftPxl, rightPxl = findPixel(i,j,shape,clone,midNose,midJaw)
	var4 = leftPxl/rightPxl


variant = [var1,var2,var3,var4,var5]
print("saved:", imagePath)
variants.append(variant)

if(len(variants) == 1):
	file = open('data.csv','a+',newline='')
	with file:
		write = csv.writer(file)
		write.writerows(variants)
	variants.clear()	

#for variant 6 i.e. calculating red circles using hough transform


# ''' Extracting the skin portion from the image '''
# # define the upper and lower boundaries of the HSV pixel
# # intensities to be considered 'skin'
# lower = np.array([0, 48, 80], dtype = "uint8")
# upper = np.array([20, 255, 255], dtype = "uint8")
# #extracting the skin portion from image 
# converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# skinMask = cv2.inRange(converted, lower, upper)
# # apply a series of erosions and dilations to the mask using an elliptical kernel
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# skinMask = cv2.erode(skinMask, kernel, iterations = 2)
# skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
# # blur the mask to help remove noise, then apply the mask to the image
# skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
# skin = cv2.bitwise_and(image, image, mask = skinMask)
# # show the skin in the image along with the mask
# cv2.imshow("skin",skin)
# cv2.waitKey(0)


# ''' applying hough transform '''
# grayImg = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
# # Blur using 3 * 3 kernel.
# gray_blurred = cv2.blur(grayImg, (3, 3))
# # Apply Hough transform on the blurred image.
# detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40)
# print(detected_circles)
# # Draw circles that are detected.
# if detected_circles is not None:
#     # Convert the circle parameters a, b and r to integers.
#     detected_circles = np.uint16(np.around(detected_circles))
#     for pt in detected_circles[0, :]:
#         a, b, r = pt[0], pt[1], pt[2]
  
#         # Draw the circumference of the circle.
#         cv2.circle(skin, (a, b), r, (0, 255, 0), 2)
  
#         # Draw a small circle (of radius 1) to show the center.
#         cv2.circle(skin, (a, b), 1, (0, 0, 255), 3)
#         cv2.imshow("Detected Circle", skin)
#         cv2.waitKey(0)


