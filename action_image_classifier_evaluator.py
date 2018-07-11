# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 19:08:52 2018

@author: CVPR
"""

# Program for evaluating an action image classification model
# Run the program with the model-name as a command-line argument, for example,
# $ python action_image_classifier_evaluator.py resnet50

from __future__ import print_function
import tensorflow as tf
import keras
from keras.models import *
from keras.applications import resnet50
from keras.models import model_from_json
import numpy as np
import cv2
from PIL import Image
from action_image_dataloader import *
import glob
import argparse

def load_model(model_name):
	# Specify json model filename
	model_json = model_name + "_model.json"
	# Specify model weights filename
	model_weights = model_name + "_mdl_best_2.h5"
	# Read json model
	json_file = open(model_json, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	# Convert json model to keras model
	loaded_model = model_from_json(loaded_model_json)
	# Load model weights
	loaded_model.load_weights(model_weights)
	# Return loaded model
	return loaded_model

def load_data(data_path):
	# Load test data
	data = action_image_dataloader(data_path, mode='Test')
	test_inputs, test_labels = data.get_data()
	return (test_inputs, test_labels)

def get_classes_and_labels(data_path):
	# Enlist class-names in the given dataset
	classes = os.listdir(data_path + "Test") #['Concert', 'Cooking', 'Craft', 'Teleshopping', 'Yoga']
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

def evaluate_on_random_data(data_path, loaded_model):
	# Get class-names in the given dataset and corresponding class-labels
	classes, labels = get_classes_and_labels(data_path)
	# Iterate over all classes
	for c in classes:
		# Load test images
		test_im1 = cv2.resize(np.array(Image.open(c + '1.jpg').convert('RGB')), (224, 224))
		test_im2 = cv2.resize(np.array(Image.open(c + '2.jpg').convert('RGB')), (224, 224))
		# Add a dummy dimension to the images to match the input dimensions of the classifier
		test_im1 = np.expand_dims(test_im1, axis=0)
		test_im2 = np.expand_dims(test_im2, axis=0)
		# Get predictions of the classifier on the test images
		prediction1 = loaded_model.predict(test_im1)
		prediction2 = loaded_model.predict(test_im2)
		# Print classifier predictions
		print("Class: " + c)
		print("\n" + c + " 1:")
		print(prediction1)
		print((prediction1 == np.max(prediction1)) * 1)
		print("\n" + c + " 2:")
		print(prediction2)
		print((prediction2 == np.max(prediction2)) * 1)
		print()

def evaluate_classwise(data_path, loaded_model):
	# Get class-names in the given dataset and corresponding class-labels
	classes, labels = get_classes_and_labels(data_path)
	# Perform class-wise evaluation
	for c in classes:
		print("\nClass " + c + ":")
		# Load test images for class c
		test_inputs = glob.glob(data_path + 'Test/' + c + '/*.*')
		# Convert all images to RGB and resize to (224, 224)
		for i in range(len(test_inputs)):
			test_inputs[i] = cv2.resize(np.array(Image.open(test_inputs[i]).convert('RGB')), (224, 224))
		# Convert input images to numpy array
		test_inputs = np.array(test_inputs)
		# Prepare labels for the images according to their class
		test_labels = np.tile(labels[c],  (100, 1))
		# Get positive classification score
		pos_score = loaded_model.evaluate(test_inputs, test_labels, batch_size=16, verbose=1)
		print(pos_score)
		# Calculate number of mis-classifications
	for neg in classes:
		if (neg == c):
			continue
		print("Negative " + neg + ":")
		neg_labels = np.tile(labels[neg], (100, 1))
		# Get mis-classification score for negative class neg
		neg_score = loaded_model.evaluate(test_inputs, neg_labels, batch_size=16, verbose=1)
		# Print mis-classification score
		print(neg_score)

def evaluate_on_test_set(test_inputs, test_labels, loaded_model):
	# Evaluate model on test set
	score = loaded_model.evaluate(test_inputs, test_labels, batch_size=16, verbose=1)
	# Print evaluation scores
	print(loaded_model.metrics_names)
	print(score)

def evaluate_model(data_path, loaded_model):
	# Load test data
	test_inputs, test_labels = load_data(data_path)
	# Evaluate model on test set
	evaluate_on_test_set(test_inputs, test_labels, loaded_model)
	# Evaluate class-wise classification accuracy of the model
	evaluate_classwise(data_path, loaded_model)
	# Evaluate model on random data
	evaluate_on_random_data(data_path, loaded_model)

def main():
	# Parse command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("model_name", type=str)
	args = parser.parse_args()
	# Get model name from command-line argument
	model_name = args.model_name
	# Load model
	loaded_model = load_model(model_name)
	# Specify data path
	data_path = "/home/rick/Downloads/Datasets/Action_Images/"	
	# Compile loaded model
	loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
	# Function-call for model evaluation
	evaluate_model(data_path, loaded_model)

if __name__ == "__main__":
	main()
