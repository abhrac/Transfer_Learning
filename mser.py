import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def detect_mser(im):
	#Create MSER object
	mser = cv2.MSER_create()
	#Convert to gray scale
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# Store a copy of the original image
	vis = im.copy()
	# Detect MSERs in the gray scale image
	regions, bboxes = mser.detectRegions(gray)
	# If no MSER is found in the image return an empty list
	if len(bboxes) == 0:
		return []
	# Print the shape of the list containing the bounding boxes
	print(bboxes.shape)
	# Get convex-hulls corresponding to the regions
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	cv2.polylines(vis, hulls, 1, (0, 255, 0))
	# Define a mask for the image
	mask = np.zeros((im.shape[0], im.shape[1], 1), dtype=np.uint8)
	# Iterate over all convex-hulls corresponding to the MSERs
	for contour in hulls:
		# Draw contours for the corresponding convex-hull
	    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
	# Mask all regions in the image except the MSERs
	mser_only = cv2.bitwise_and(im, im, mask=mask)
	# Return the masked image containing only the MSERs
	return mser_only
