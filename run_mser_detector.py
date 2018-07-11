# Program for extracting MSERs from a given image dataset
# Run the program as:
# $ python run_mser_detector.py -split train/test
# If no argument is given, "train" will be taken by default

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
	# Iterate over all the images in the given class
	for (i, im) in enumerate(images):
		# Convert each image to RGB
		im = np.array(Image.open(im).convert('RGB'))
		# Get image containing only the MSERs
		im = detect_mser(im)
		if im != []:
			# If the image has atleat one MSER, save it to the destination directory
			cv2.imwrite(destination_path + image_class + '/' + image_class + '_msers_' + split + '_' + str(i) + '.png', im)
			print("Saved " + image_class + '_mser_' + split + '_' + str(i) + '.png')

def extract_msers_from_dataset(data_path, destination_path, split="Train"):
	# Specify the data path for the corresponding split
	data_path = data_path + split + "/"
	# Specify the destination path for the corresponding split
	destination_path = destination_path + split + "/"
	# Specify the classes to be considered
	classes = ["Concert", "Cooking", "Craft", "Teleshopping"]
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
	data_path = "/home/rick/Downloads/Datasets/Action_Images/"
	# Get dataset split from command-line argument
	split = args.split
	# Specify destination path
	destination_path = "MSERs/"
	# Function-call for extracting MSERs from the given dataset
	extract_msers_from_dataset(data_path, destination_path, split)

if __name__ == "__main__":
	main()
