from imutils import face_utils, resize
import numpy as np
import dlib
import cv2
from shapely.geometry import Point,Polygon


# capture the photo on which
# the image processing will happen
cam = cv2.VideoCapture(0)
cv2.namedWindow("Test")
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "test.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

cam.release()

cv2.destroyAllWindows()

variants = []

# this function generate the number of pixels which is present inside 
# the contour of given facial landmarks

# this happens via the use of shapely which creates a boundary on that facial landmark
# and we use the bounds of the pixels generated via dlib to create a rectangular region 
# and iterating over it and checking with a simple pass that if point lies inside the contour
# the pixels which are inside are used for counting and their alpha, beta values of landmark 
# in LAB color space.  
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

	cv2.imshow("check",clone)
	cv2.waitKey(0)
	return (pixelsNum,totalAlpha,totalBeta)

def find_pixels(i,j,shape,clone):	
	# size = j-i+1
	coords = []
	# x and y are interchanged
	for (y, x) in shape[i:j]:
		coords.append((x,y))
	p1 = Polygon(coords)
	return numpixels(p1,clone)


# this is the same function as above, we just obtained the number of pixels
# in right side and left side of the face by using the median pixel in the jaw and nose
# and checking if the pixel is inside the contour of face and for left and right, checking the 
# value of line if it is greater than 0 or not
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

	cv2.imshow("check",clone)
	cv2.waitKey(0)
	return (leftNum, rightNum)

	



# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
imagePath = 'test.png'
image = cv2.imread(imagePath)
circleImage = image.copy()
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
	clonecopy = clone.copy()
	#Required
		# alpha(eyes)= alpha(leftEye)+alpha(rightEye), numPixels(eyes)= numPixels(leftEye)+numPixels(rightEye)
		# beta(eyes)= beta(leftEye)+beta(rightEye) 
		# alpha(lips)= alpha(mouth), numPixels(lips)=numPixels(mouth)

	# Given
		# ([('mouth', (48, 68)), ('inner_mouth', (60, 68)), ('right_eyebrow', (17, 22)), 
		# ('left_eyebrow', (22, 27)), ('left_eye', (36, 42)), ('right_eye', (42, 48)),
		# ('nose', (27, 36)), ('jaw', (0, 17))])

	# # test purposes (individual parts)
	copy1 = image.copy()	
	for (x, y) in shape[i:j]:
		copy1[y,x] = (255,255,255)
		# cv2.circle(copy1, (x, y), 1, (0, 0, 255), -1)
	cv2.imshow('image',copy1)
	cv2.waitKey(0)

	# for Left eye
	(i,j) = (36,42)
	numLeye, alphaLeye, betaLeye = find_pixels(i,j,shape,clone)

	# for Right eye
	(i,j) = (42,48)
	cloneEye = clonecopy.copy()
	numReye, alphaReye, betaReye = find_pixels(i,j,shape,cloneEye)

	numEye = numLeye+numReye
	alphaEye = alphaLeye+alphaReye
	betaEye = betaLeye+betaReye

	var1 = alphaEye/numEye
	var2 = betaEye/numEye
	var5 = numLeye/numReye

	#for mouth
	(i,j) = (48,68)
	cloneMouth = clonecopy.copy()
	numMouth, alphaMouth, betaMouth = find_pixels(i,j,shape,cloneMouth)

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
	cloneJaw = clonecopy.copy()
	cv2.circle(cloneJaw,(midNose[1],midNose[0]),1,(255,255,0),-1)
	cv2.circle(cloneJaw,(midJaw[1],midJaw[0]),1,(255,0,0),-1)
	cv2.imshow("check",cloneJaw)
	cv2.waitKey(0)


	# for variant 4
	clonePxl = clonecopy.copy()
	leftPxl, rightPxl = findPixel(i,j,shape,clonePxl,midNose,midJaw)
	var4 = leftPxl/rightPxl

	#for variant 6
	gray = cv2.cvtColor(circleImage, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)

	for (i, rect) in enumerate(rects):	
		shape2 = predictor(gray, rect)
		shape2 = face_utils.shape_to_np(shape2)
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		circleImage = circleImage[y:y+h,x:x+w]
		lab = cv2.cvtColor(circleImage,cv2.COLOR_BGR2LAB)
		L,A,B=cv2.split(lab)
		''' applying hough transform '''
		grayImg = cv2.cvtColor(circleImage, cv2.COLOR_BGR2GRAY)
		# Blur using 3 * 3 kernel.
		A_blurred = cv2.blur(A, (3,3))
		# for low image, may have wrong results
		gray_blurred = cv2.blur(grayImg, (3, 3))
		# Apply Hough transform on the blurred image.
		detected_circles = cv2.HoughCircles(A_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 20, param2 = 10, minRadius = 1, maxRadius = 4)
		# Draw circles that are detected.
		if detected_circles is not None:
			# Convert the circle parameters a, b and r to integers.
			detected_circles = np.uint16(np.around(detected_circles))
			for pt in detected_circles[0, :]:
				a, b, r = pt[0], pt[1], pt[2]
				cv2.circle(lab, (a, b), r, (0, 255, 0), 2)
				var6 += 1
				cv2.imshow("Detected Circle", lab)
				cv2.waitKey(0)

variant = [var1,var2,var3,var4,var5,var6]
print(variant)

# obtained through training data 
var_mean = [133.527,132.528,132.29,1.338,1.012,2.871]
var_std = [5.158,7.104,11.581,0.2275,0.274,3.35]
to_display = ["Scleritis, Subconjunctival Hemorrhage, Corneal Ulcer, Extraocular Muscle Entrapment (Inf Rectus), Muddy Brown Sclera, Periorbital Cellulitis, Periorbital Echymosis", "Icterus", "Cyanosis", "Central CN 7 Palsy, Cervical Adenopathy, Parotitis, Peripheral CN7 Palsy, Submandibular Abscess","Central CN 7 Palsy, Peripheral CN7 Palsy, Extraocular Muscle Entrapment (Inf Rectus), Horner’s Syndrome, Periorbital Cellulitis, Periorbital Echymosis","Acnes, Hematoma of the Scalp with Cellulitis, Zoster and Cellulitis"]

t = 1.5
count = 0

if variant[0] > var_mean[0]+(t*var_std[0]):
	count += 1
	print("Anamolous according to variant 1")
	print("Can have: ", to_display[0])

if variant[1] > var_mean[1]+(t*var_std[1]):
	count += 1
	print("Anamolous according to variant 2")
	print("Can have: ", to_display[1])

if variant[2] < var_mean[2]+(t*var_std[2]):
	count += 1
	print("Anamolous according to variant 3")
	print("Can have: ", to_display[2])

if variant[3] > var_mean[3]+(t*var_std[3]) or variant[3] > var_mean[3]-(t*var_std[3]):
	count += 1
	print("Anamolous according to variant 4")
	print("Can have: ", to_display[3])

if variant[4] > var_mean[4]+(t*var_std[4]) or variant[4] > var_mean[4]-(t*var_std[4]):
	count += 1
	print("Anamolous according to variant 5")
	print("Can have: ", to_display[4])

if variant[5] > var_mean[5]+(t*var_std[5]):
	count += 1
	print("Anamolous according to variant 6")
	print("Can have: ", to_display[5])

if count == 0:
	print("No anamolous variant detected")