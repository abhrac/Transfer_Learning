from __future__ import print_function
import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.layers.core import*
from keras.models import model_from_json
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from plot_losses import *
from keras_callbacks import *
from action_image_dataloader import *

def train_combiner(train_inputs, train_labels, test_inputs, test_labels):
	# Create PlotLosses object for plotting training loss
	plot_losses = PlotLosses()
	# Specify batch-size
	batch_size = 1024
	# Specifiy number of epochs
	num_epochs = 400
	# Describe model architecture
	inp = Input(shape=(11, ))
	x = Dense(16, activation='relu')(inp)
	predictions = Dense(5, activation='softmax')(x)
	# Create Model object
	model = Model(inputs=inp, outputs=predictions)
	# Compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# Train model
	model.fit(x=train_inputs, y=train_labels, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True,
				validation_data=(test_inputs, test_labels),
				callbacks=[plot_losses]+callbacks('combiner_mdl_best.h5'))
	# Save the model as a json file
	model_json = model.to_json()
	with open("combiner_model.json", "w") as json_file:
		json_file.write(model_json)
	# Display plot for training losses
	plt.show()

def main():
	# Load train set
	train_inputs = np.load("Train_pred_features.npy")
	train_labels = np.load("Train_labels.npy")
	# Load test set
	test_inputs = np.load("Test_pred_features.npy")
	test_labels = np.load("Test_labels.npy")
	# Remove single dimensional second axis from train and test set
	train_inputs = np.squeeze(train_inputs, axis=1)
	test_inputs = np.squeeze(test_inputs, axis=1)
	print(train_inputs.shape, " ", test_inputs.shape)
	# Function-call for training the model-combiner
	train_combiner(train_inputs, train_labels, test_inputs, test_labels)

if __name__ == "__main__":
	main()
