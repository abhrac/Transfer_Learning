# Program for extracting MSERs from a given image dataset
# Run the program as:
# $ python run_mser_detector.py -split Train/Test
# If no argument is given, "train" will be taken by default

from __future__ import print_function
from mser import *
import numpy as np
import argparse
import os
import glob
import random
import cv2
from PIL import Image

def extract_msers_from_class(data_path, image_class, destination_path, split):
	# Read all images of the given class
	images = glob.glob(data_path + image_class + '/*')
	print(os.path.exists(data_path + image_class))
	print(len(images))
	# Iterate over all the images in the given class
	for (i, im) in enumerate(images):
		# Convert each image to RGB
		im = np.array(Image.open(im).convert('RGB'))
		# Get image containing only the MSERs
		im = detect_mser(im)
		if im != []:			
			# Set destination directory
			dest_dir = destination_path + image_class + '/'
			# If destination directory does not exist, create it
			if not os.path.exists(dest_dir):
				os.makedirs(dest_dir)
			# If the image has atleat one MSER, save it to the destination directory
			cv2.imwrite(dest_dir + image_class + '_msers_' + split + '_' + str(i) + '.png', im)
			print("Saved " + image_class + '_mser_' + split + '_' + str(i) + '.png')

def get_classes_and_labels(data_path):
	# Enlist class-names in the given dataset
	classes = os.listdir(data_path) #['Concert', 'Cooking', 'Craft', 'Teleshopping', 'Yoga']
	# Sort class-names in lexicographical order
	classes.sort()
	# Declare an empty dictionary for class labels
	labels = {}
	# Prepare labels for each class
	for i, c in enumerate(classes):
		labels[c] = np.array([1 if (j==i) else 0 for j in range(len(classes))])
	print(labels)
	# Return class-names and labels
	return (classes, labels)

def extract_msers_from_dataset(data_path, destination_path, split="Train"):
	# Specify the data path for the corresponding split
	data_path = data_path + split + "/"
	# Specify the destination path for the corresponding split
	destination_path = destination_path + split + "/"
	# Specify the classes to be considered
	classes, _ = get_classes_and_labels(data_path)
	# Iterate over all the specified classes
	for c in classes:
		print("Extracting MSERs from " + c + " images...")
		# Function-call for extracting MSERs from images of class c
		extract_msers_from_class(data_path, c, destination_path, split)

def main():
	# Parse command-line argument
	parser = argparse.ArgumentParser()
	parser.add_argument('-split', type=str, required=False, default='Train')
	args = parser.parse_args()
	# Specify dataset path
	data_path = "Stanford_40_Actions/"
	# Get dataset split from command-line argument
	split = args.split
	# Specify destination path
	destination_path = "MSERs/"
	# Function-call for extracting MSERs from the given dataset
	extract_msers_from_dataset(data_path, destination_path, split)

if __name__ == "__main__":
	main()
