from __future__ import print_function
import numpy as np
import scipy.io

def convert_to_numpy_array(file_name, num_ex, num_classes):
	# Read mat file
	mat_file = scipy.io.loadmat(file_name)
	# Get file-name without extension
	arr_name, _ = file_name.split('.')
	# Extract array from mat file
	np_arr = mat_file[arr_name]
	# Set start index of first valid slice
	start_idx = 2
	# Declare a list to store the pruned array
	new_arr = []
	# Iterate over the original array
	for i in range(0, num_classes):
		# Set end index of ith valid slice
		end_idx = start_idx + num_ex
		# Extract ith valid slice
		arr_slice = np_arr[start_idx:end_idx, :]
		# Append the slice to the array containing all previous slices
		new_arr.append(arr_slice)
		# Set start index of next valid slice
		start_idx = end_idx + 2
	# Convert the pruned-array list to a numpy array
	new_arr = np.array(new_arr)
	# Reshape new_arr
	new_arr = new_arr.reshape(-1, new_arr.shape[-1])
	# Save the pruned array
	np.save((arr_name + '.npy'), new_arr)
	print(new_arr.shape)

def main():
	# Set file-name
	file_name = 'action_image_plane_features_train.mat'
	# Set number of classes
	num_classes = 5
	# Set number of examples in each class
	num_ex = 40
	# Function call for converting the given mat file to a numpy array
	convert_to_numpy_array(file_name, num_ex, num_classes)

if __name__ == '__main__':
	main()
