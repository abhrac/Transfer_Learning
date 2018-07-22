import keras.backend as K
import numpy as np
from scipy.misc import imsave
from keras.models import model_from_json
import matplotlib.pyplot as plt
from PIL import Image

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

# Utility function to convert a tensor into a valid image
def deprocess_image(x):
    # Normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def visualize_mser_classifier():
	model = load_model('mser_classifier')
	# Get the symbolic outputs of each "key" layer
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	layer_name = 'block5_conv3'
	# Can be any integer from 0 to 511, as there are 512 filters in that layer
	kept_filters = []
	for filter_index in range(200):
		print(filter_index)
		# Loss function that maximizes the activation
		# of the nth filter of the layer considered
		layer_output = layer_dict[layer_name].output
		loss = K.mean(layer_output[:, :, :, filter_index])
		# Placeholder for the input images
		input_img = model.input
		# Dimensions of the generated pictures for each filter.
		img_width = 128
		img_height = 128
		# Compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]
		# Normalize the gradient
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
		# This function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])
		# Step size for gradient ascent
		step = 1
		# Start with a random gray image
		input_img_data = np.random.random((1, img_width, img_height, 3)) * 20 + 128
		# run gradient ascent for 20 steps
		for i in range(20):
		    loss_value, grads_value = iterate([input_img_data])
		    input_img_data += grads_value * step
		    print('Current loss value:', loss_value)
		    if loss_value <= 0.:
		    	# some filters get stuck to 0, we can skip them
		    	break
		# Append generated image
		if loss_value > 0:
			img = deprocess_image(input_img_data[0])
			kept_filters.append((img, loss_value))
	# Stich the best 16 filters on a 4 x 4 grid
	n = 4
	# The filters that have the highest loss are assumed to be better-looking.
	# Keep the best 64 filters.
	kept_filters.sort(key=lambda x: x[1], reverse=True)
	kept_filters = kept_filters[:n * n]
	# Build a black picture with enough space for
	# all 4 x 4 filters of size 128 x 128, with a 5px margin in between
	margin = 5
	width = n * img_width + (n - 1) * margin
	height = n * img_height + (n - 1) * margin
	stitched_filters = np.zeros((width, height, 3))
	# Fill the picture with the saved filters
	for i in range(n):
	    for j in range(n):
	        img, loss = kept_filters[i * n + j]
	        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
	                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
	# Save the result to disk
	imsave('mser_classifier_stitched_filters_%dx%d.png' % (n, n), stitched_filters)

def visualize_face_classifier():
	model = load_model('face_classifier')
	# Get the symbolic outputs of each "key" layer
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	layer_name = 'block5_conv3'
	# Can be any integer from 0 to 511, as there are 512 filters in that layer
	kept_filters = []
	for filter_index in range(200):
		print(filter_index)
		# Loss function that maximizes the activation
		# of the nth filter of the layer considered
		layer_output = layer_dict[layer_name].output
		loss = K.mean(layer_output[:, :, :, filter_index])
		# Placeholder for the input images
		input_img = model.input
		# Dimensions of the generated pictures for each filter.
		img_width = 128
		img_height = 128
		# Compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]
		# Normalize the gradient
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
		# This function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])
		# Step size for gradient ascent
		step = 1
		# Start with a random gray image
		input_img_data = np.random.random((1, img_width, img_height, 3)) * 20 + 128
		# run gradient ascent for 20 steps
		for i in range(20):
		    loss_value, grads_value = iterate([input_img_data])
		    input_img_data += grads_value * step
		    print('Current loss value:', loss_value)
		    if loss_value <= 0.:
		    	# some filters get stuck to 0, we can skip them
		    	break
		# Append generated image
		if loss_value > 0:
			img = deprocess_image(input_img_data[0])
			kept_filters.append((img, loss_value))
	# Stich the best 16 filters on a 4 x 4 grid
	n = 4
	# The filters that have the highest loss are assumed to be better-looking.
	# Keep the best 64 filters.
	kept_filters.sort(key=lambda x: x[1], reverse=True)
	kept_filters = kept_filters[:n * n]
	# Build a black picture with enough space for
	# all 4 x 4 filters of size 128 x 128, with a 5px margin in between
	margin = 5
	width = n * img_width + (n - 1) * margin
	height = n * img_height + (n - 1) * margin
	stitched_filters = np.zeros((width, height, 3))
	# Fill the picture with the saved filters
	for i in range(n):
	    for j in range(n):
	        img, loss = kept_filters[i * n + j]
	        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
	                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
	# Save the result to disk
	imsave('face_classifier_stitched_filters_%dx%d.png' % (n, n), stitched_filters)

def visualize_resnet50_classifier():
	model = load_model('resnet50')
	# Get the symbolic outputs of each "key" layer
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	layer_name = 'conv1'
	# Can be any integer from 0 to 511, as there are 512 filters in that layer
	kept_filters = []
	for filter_index in range(64):
		print(filter_index)
		# Loss function that maximizes the activation
		# of the nth filter of the layer considered
		layer_output = layer_dict[layer_name].output
		loss = K.mean(layer_output[:, :, :, filter_index])
		# Placeholder for the input images
		input_img = model.input
		# Dimensions of the generated pictures for each filter.
		img_width = 128
		img_height = 128
		# Compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]
		# Normalize the gradient
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
		# This function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])
		# Step size for gradient ascent
		step = 1
		# Start with a random gray image
		input_img_data = np.random.random((1, img_width, img_height, 3)) * 20 + 128
		# run gradient ascent for 20 steps
		for i in range(50):
		    loss_value, grads_value = iterate([input_img_data])
		    input_img_data += grads_value * step
		    print('Current loss value:', loss_value)
		    if loss_value <= 0.:
		    	# some filters get stuck to 0, we can skip them
		    	break
		# Append generated image
		if loss_value > 0:
			img = deprocess_image(input_img_data[0])
			kept_filters.append((img, loss_value))
	# Stich the best 16 filters on a 5 x 5 grid
	n = 5
	# The filters that have the highest loss are assumed to be better-looking.
	# Keep the best 64 filters.
	kept_filters.sort(key=lambda x: x[1], reverse=True)
	kept_filters = kept_filters[:n * n]
	# Build a black picture with enough space for
	# all 5 x 5 filters of size 128 x 128, with a 5px margin in between
	margin = 5
	width = n * img_width + (n - 1) * margin
	height = n * img_height + (n - 1) * margin
	stitched_filters = np.zeros((width, height, 3))
	# Fill the picture with the saved filters
	for i in range(n):
	    for j in range(n):
	        img, loss = kept_filters[i * n + j]
	        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
	                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
	# Save the result to disk
	imsave('resnet50_stitched_filters_%dx%d.png' % (n, n), stitched_filters)

def main():
	visualize_resnet50_classifier()
	visualize_mser_classifier()
	visualize_face_classifier()

if __name__ == '__main__':
	main()
