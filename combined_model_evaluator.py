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

def evaluate_on_random_data(data_path):
	# Load models
	basis_model = load_model("resnet50")
	mser_model = load_model("mser_classifier")
	face_model = load_model("face_classifier")
	combiner_model = load_model("combiner")
	# Get class-names in the given dataset and corresponding class-labels
	classes, labels = get_classes_and_labels(data_path)
	test_inputs = np.load("Test_pred_features.npy")
	# Iterate over all classes
	for c in classes:
		print("Class: " + c)
		im1 = np.array(Image.open(c + '1.png').convert('RGB'))
		im2 = np.array(Image.open(c + '2.png').convert('RGB'))
		test_images = [im1, im2]
		for (i, im) in enumerate(test_images):
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
			# Get predictions of the classifier on the test images
			prediction = combiner_model.predict(probabilities)
			# Print classifier predictions
			print("\n" + c + " " + str(i), ":")
			print(prediction)
			print((prediction == np.max(prediction)) * 1)
			print()

def evaluate_classwise(data_path):
	basis_model = load_model("resnet50")
	mser_model = load_model("mser_classifier")
	face_model = load_model("face_classifier")
	combiner_model = load_model("combiner")
	basis_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
	mser_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
	face_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
	combiner_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
	# Get class-names in the given dataset and corresponding class-labels
	classes, labels = get_classes_and_labels(data_path)
	pred_features = np.load("Test_pred_features.npy")
	# Perform class-wise evaluation
	for (i, c) in enumerate(classes):
		print("\nClass " + c + ":")
		probabilities = pred_features[(i*100):(i+1)*100, :]
		probabilities = np.squeeze(probabilities, axis=1)
		# Get number of images
		num_images = 100
		# Convert input images to numpy array
		#test_inputs = np.array(test_inputs)
		# Prepare labels for the images according to their class
		test_labels = np.tile(labels[c],  (num_images, 1))
		# Get positive classification score
		pos_score = combiner_model.evaluate(probabilities, test_labels, batch_size=100, verbose=1)
		print(pos_score)
		# Calculate number of mis-classifications
		for neg in classes:
			if (neg == c):
				continue
			print("Negative " + neg + ":")
			neg_labels = np.tile(labels[neg], (num_images, 1))
			# Get mis-classification score for negative class neg
			neg_score = combiner_model.evaluate(probabilities, neg_labels, batch_size=100, verbose=1)
			# Print mis-classification score
			print(neg_score)
		print()

def main():
	# Specify data path
	data_path = "Action_Images/"
	# Evaluate model on random data
	evaluate_on_random_data(data_path)
  # Evaluate model class-wise
	evaluate_classwise(data_path)

if __name__ == "__main__":
	main()
