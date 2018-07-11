import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.layers.core import*
from keras.preprocessing.image import *
from keras.applications import resnet50
from keras.models import model_from_json
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

def get_mser_classifier():
	# Sequential model description
	model = Sequential()
	# Conv1 - Input
	model.add(Conv2D(96, (11, 11), input_shape=(224, 224, 3), strides=(4, 4), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	model.add(BatchNormalization())
	# Conv2
	model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	model.add(BatchNormalization())
	# Conv3
	model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	# Conv4
	model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	# Conv5
	model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	# Dropout
	model.add(Dropout(rate=0.5))
	# FC6
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	# Dropout
	model.add(Dropout(rate=0.5))
	# FC7
	model.add(Dense(1024, activation='relu'))
	# FC8 - Output
	model.add(Dense(4, activation='softmax'))
	# Return the described model
	return model

def train_mser_based_classifier(train_data_path, test_data_path):
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
	# Create object for the new model
	model = get_mser_classifier()
	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# Train the final layers of the model
	model.fit(x=train_inputs, y=train_labels, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True, validation_data=(test_inputs, test_labels), callbacks=[plot_losses]+callbacks('mser_classifier_mdl_best.h5'))
	# model.fit_generator(generator, epochs=num_epochs) #, verbose=1, shuffle=True, validation_data=(test_inputs, test_labels)), callbacks=[plot_losses]+callbacks('mser_classifier_mdl_best.h5'))
	# Save the model as a json file
	model_json = model.to_json()
	with open("mser_classifier_model.json", "w") as json_file:
		json_file.write(model_json)
	# Save model weights
	model.save_weights("mser_classifier_model.h5")
	# Display plot training losses
	plt.show()

def main():
	# Specify train set path
	train_data_path = "MSERs/"
	# Specify test set path
	test_data_path = "MSERs/"
	# Function-call for training and validating the mser-based classifier on the given dataset
	train_mser_based_classifier(train_data_path, test_data_path)

if __name__ == "__main__":
	main()
