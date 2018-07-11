# Program for extracting faces from a given image dataset
# Run the program as:
# $ python run_mser_detector.py -split train/test
# If no argument is given, "train" will be taken by default

from face_detector_facenet import *
import numpy as np
import os
import argparse
import glob
import random
import cv2
from PIL import Image

def extract_faces_from_class(data_path, image_class, destination_path, split):
	# Read all images of the given class
	images = glob.glob(data_path + image_class + '/*')
	# Iterate over all the images in the given class
	for (i, im) in enumerate(images):
		# Convert each image to RGB and resize to (224, 224)
		im = cv2.resize(np.array(Image.open(im).convert('RGB')), (224, 224))
		# Get image containing only the faces
		im = detect_faces_facenet(im)
		if im != []:
			# If the image has atleat one face, save it to the destination directory
			Image.fromarray(im).save(destination_path + image_class + '/' + image_class + '_faces_' + split + '_' + str(i) + '.png')
			print("Saved " + image_class + '_faces_' + split + '_' + str(i) + '.png')

def extract_faces_from_dataset(data_path, destination_path, split="Train"):
	# Specify the data path for the corresponding split
	data_path = data_path + split + "/"
	# Specify the destination path for the corresponding split
	destination_path = destination_path + split + "/"
	# Specify the classes to be considered
	classes = ["Concert", "Teleshopping"]
	# Iterate over all the specified classes
	for c in classes:
		print("Extracting faces from " + c + " images...")
		# Function-call for extracting faces from images of class c
		extract_faces_from_class(data_path, c, destination_path, split)

def main():
	# Parse command-line argument
	parser = argparse.ArgumentParser()
	parser.add_argument('-split', type=str, required=False, default='Train')
	args = parser.parse_args()
	# Specify dataset path
	data_path = "Action_Images/"
	# Get dataset split from command-line argument
	split = args.split
	# Specify destination path
	destination_path = "Faces/"
	# Function-call for extracting faces from the given dataset
	extract_faces_from_dataset(data_path, destination_path, split)

if __name__ == "__main__":
	main()
