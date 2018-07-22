import numpy as np
import argparse
from keras.models import model_from_json

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

def show_layers(model):
	model = load_model(model)
	for i, layer in enumerate(model.layers):
		print(i, layer.name, layer.output)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('model', type=str)
	args = parser.parse_args()
	model = args.model
	show_layers(model)

if __name__ == '__main__':
	main()
