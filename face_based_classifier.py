import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.layers.core import*
from keras.preprocessing.image import *
from keras.applications import vgg16
from keras.models import model_from_json
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
import random
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from plot_losses import *
from keras_callbacks import *
from action_image_dataloader import *

def fine_tune_vgg16(generator, train_inputs, train_labels, test_inputs, test_labels, num_epochs, batch_size, plot_losses):
	# Loading the best model from previous step
	json_file = open('face_classifier_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("face_classifier_mdl_best_1.h5")
	# Renaming the model for convenience
	model = loaded_model
	# Freeze the first 169 layers of the model
	# Only the later layers will be trainable
	for layer in model.layers[:17]:
		layer.trainable = False
	for layer in model.layers[17:]:
		layer.trainable = True
	# Compile the model
	model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
	# Fine-tune the model wih a slow laerning rate and a non-adaptive optimization algorithm in-order
	# to prevent massive gradient updates from wercking the previously learned weights
	model.fit(x=train_inputs, y=train_labels, batch_size=batch_size, epochs=num_epochs, verbose=1, 
		  shuffle=True, validation_data=(test_inputs, test_labels), 
		  callbacks=[plot_losses]+callbacks('face_classifier_mdl_best_2.h5'))
	# model.fit_generator(generator, epochs=num_epochs, verbose=1,
	# shuffle=True, validation_data=(test_inputs, test_labels),
	# callbacks=[plot_losses]+callbacks('face_classifier_mdl_best_2.h5'))
	# Save the fine-tuned model as a json file
	model_json = model.to_json()
	with open("face_classifier_model.json", "w") as json_file:
		json_file.write(model_json)
	# Save model weights
	model.save_weights("face_classifier_model.h5")
	# Display plot training losses
	plt.show()

def train_vgg16(train_data_path, test_data_path):
	# Load train set
	data = action_image_dataloader(train_data_path)
	train_inputs, train_labels = data.get_data()
	print(train_inputs.shape)
	print(train_labels.shape)
	# Load test set
	test_data = action_image_dataloader(test_data_path, mode='Test')
	test_inputs, test_labels = test_data.get_data()
	print(test_inputs.shape)
	print(test_labels.shape)
	# Create ImageDataGenerator object for data-augmentation
	datagen = ImageDataGenerator(
				rotation_range = 40, 
				width_shift_range = 0.2,
				height_shift_range = 0.2,
				shear_range = 0.2,
				zoom_range = 0.2,
				horizontal_flip = True,
				fill_mode = 'nearest')
	# Create PlotLosses object for plotting training loss
	plot_losses = PlotLosses()
	# Specify batch-size
	batch_size = 2
	# Specifiy number of epochs
	num_epochs = 2
	# Create data-generator
	generator  = datagen.flow(train_inputs, train_labels, batch_size=batch_size)
	# Load base-model VGG16
	base_model = vgg16.VGG16(weights='imagenet', include_top=False)
	# Define the final layers of the model
	x = base_model.output
	x = Dropout(rate=0.5)(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(2, activation='softmax')(x)
	# Create object for the new model
	model = Model(inputs=base_model.input, outputs=predictions)
	# Freeze all the layers of the base-model
	# Only the newly defined final layers will be trainable
	for layer in base_model.layers:
		layer.trainable = False
	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# Train the final layers of the model
	model.fit(x=train_inputs, y=train_labels, batch_size=batch_size, epochs=num_epochs, verbose=1, 
		  shuffle=True, validation_data=(test_inputs, test_labels), 
		  callbacks=[plot_losses]+callbacks('face_classifier_mdl_best_1.h5'))
	# model.fit_generator(generator, epochs=num_epochs, verbose=1, shuffle=True, 
	# validation_data=(test_inputs, test_labels), callbacks=[plot_losses]+callbacks('face_classifier_mdl_best_1.h5'))
	# Save the model as a json file
	model_json = model.to_json()
	with open("face_classifier_model.json", "w") as json_file:
		json_file.write(model_json)
	# Function call for fine-tuning a previous layer of the model
	fine_tune_vgg16(generator, train_inputs, train_labels, test_inputs, test_labels, num_epochs, batch_size, plot_losses)

def main():
	# Specify train set path
	train_data_path = "Faces/"
	# Specify test set path
	test_data_path = "Faces/"	
	# Function-call for training and validating the model on the given dataset
	train_vgg16(train_data_path, test_data_path)

if __name__ == "__main__":
	main()
