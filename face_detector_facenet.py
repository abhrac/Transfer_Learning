from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
import tensorflow as tf
import os
from facenet.src import align
import facenet.src.align.detect_face
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def extract_faces(im, bounding_boxes):
	# If there are no bounding-boxes, return an empty list
	if len(bounding_boxes) == 0:
		print("---No faces found---")
		return ([])
	mask = np.zeros(im.shape, dtype=np.uint8)
	# Iterate over the list of bounding-boxes
	for i, (x1, y1, x2, y2, acc) in enumerate(bounding_boxes):
		w = x2-x1
		h = y2-y1
		# Round-off the bounding box co-ordinates to integers
		(x1, x2, y1, y2) = map(lambda x: int(x), (x1, x2, y1, y2))
		print("Bounding box co-ordinates x1, y1, x2, y2: ")
		print (x1, y1, x2, y2)
		# Create a mask containing 1s only in the bounding-box region
		mask[y1:y2, x1:x2, :] = 1
	# Return masked image containing only faces
	return (im * mask)

def detect_faces_facenet(im):
	# Minimum size of face
	minsize = 50
	# Three steps's threshold
	threshold = [ 0.6, 0.7, 0.7 ]
	# Scale-factor
	factor = 0.709
	print('Creating networks and loading parameters')
	with tf.Graph().as_default():
	    sess = tf.Session()
	    with sess.as_default():
	        pnet, rnet, onet = align.detect_face.create_mtcnn(
	            sess, None)
	    # Run detect_face from the facenet library
	    bounding_boxes, _ = align.detect_face.detect_face(
	            im, minsize, pnet,
	            rnet, onet, threshold, factor)
	    # Return image containing only the extracted faces
	    return extract_faces(im, bounding_boxes)
