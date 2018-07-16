# Dataset preparation for training the model-combining neural network

from __future__ import print_function
from mser import *
from face_detector_facenet import *
from action_image_dataloader import *
import numpy as np
import os
import glob
import random
import cv2
from PIL import Image
import tensorflow as tf
import keras
from keras.models import *
from keras.models import model_from_json
import glob

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

def load_data(data_path, mode="Train", resize_images=True):
	# Load test data
	data = action_image_dataloader(data_path, mode, resize_images)
	test_inputs, test_labels = data.get_data()
	return (test_inputs, test_labels)

def get_predictions(inputs, labels):
	# Load basis, mser-based and face-based classifiers
	basis_model = load_model("resnet50")
	mser_model = load_model("mser_classifier")
	face_model = load_model("face_classifier")
	# Declare an empty list for storing the predictions
	pred_features = []
	# Iterate over all the images
	for (i, im) in enumerate(inputs):
		# Convert each image to RGB
		im = np.array(Image.open(im).convert('RGB'))
		# Set default predictions in case msers/faces are not present
		mser_pred = np.zeros((1, 4))
		face_pred = np.zeros((1, 2))
		# Get prediction from basis classifier
		basis_pred = basis_model.predict(np.expand_dims(cv2.resize(im, (224, 224)), axis=0))
		# Get image containing only MSERs
		msers = detect_mser(im)
		if msers != []:
			# Get predictions from mser-based classifier
			mser_pred = mser_model.predict(np.expand_dims(cv2.resize(msers, (224, 224)), axis=0))
		# Get image containing only faces
		faces = detect_faces_facenet(im)
		if faces != []:
			# Get predictions from face-based classifier
			face_pred = face_model.predict(np.expand_dims(cv2.resize(faces, (224, 224)), axis=0))
		# Append predictions from all 3 models
		probabilities = np.expand_dims(np.append(basis_pred, np.append(mser_pred, face_pred)), axis=0)
		# Append the new set of features to pred_features
		pred_features.append(probabilities)
	# Convert pred_features to a numpy array
	pred_features = np.array(pred_features)
	# Save pred_features
	np.save("pred_features.npy", pred_features)
	print(pred_features.shape)
	# Save the numpy array containing labels
	np.save("labels.npy", labels)

def main():
	data_path = "/home/rick/Downloads/Datasets/Action_Images/"
	train_inputs, train_labels = load_data(data_path, resize_images=False)
	get_predictions(train_inputs, train_labels)

if __name__ == "__main__":
	main()
