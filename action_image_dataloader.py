from __future__ import print_function
import numpy as np
import os
import glob
import random
import cv2
from PIL import Image

class action_image_dataloader():
	def __init__(self, data_path, mode='Train', resize_images=True):
		# Enlist the classes present in the dataset
		classes = os.listdir(data_path + mode + '/')
		# sort classes by their names
		classes.sort()
		# Declare a list for storing input images
		self.inputs = []
		# Declare a dummy numpy array for defining the dimensions for the data labels
		self.labels = np.zeros((1, len(classes)))
		# Iterate over every image class
		for i, c in enumerate(classes):
			# read all images in the class c
			images = glob.glob(data_path + mode + '/' + c + '/*.*')
			# append the new set of images to the input list
			self.inputs.extend(images)
			# define a label for the class c
			label_arr = np.array([1 if (j==i) else 0 for j in range(len(classes))])
			# append required number of labels to the labels list
			self.labels = np.vstack((self.labels, np.tile(label_arr,  (len(images), 1))))
			# Print the class-name and the corresponding label
			print(c, label_arr)
		# Remove the initial dummy label
		self.labels = self.labels[1:, :]
		# Return, if images are not to be resized
		if (resize_images == False):
			#self.inputs = self.inputs[:10]
			return
		# Convert every image in the training set to RGB and resize them to (224, 224)
		for i in range(len(self.inputs)):
			#if i >= 10:
			#	break
			self.inputs[i] = cv2.resize(np.array(Image.open(self.inputs[i]).convert('RGB')), (224, 224))
		# Convert the list containing the input images to a numpy array
		self.inputs = np.array(self.inputs) #[:10])

	def get_data(self):
		return (self.inputs, self.labels) #[:10, :])
	
	def __len__(self):
		return(self.inputs.shape[0])
