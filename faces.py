import math
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from scipy.misc import imsave as save

###################################################################################################
#FUNCTIONS FOR ASSIGNMENT OUTPUTS
###################################################################################################

def get_pics(number_pics, name, possible):
	'''
	Function: generates a matrix of flattened image vectors
	Inputs: 
		number_pics: the number of pictures required (int)
		name: the name of the actor (string)
		possible: a vector containing the possible indicies of the images to be accessed
		Note that the images are saved in the form "Butler57.jpg" where 57 is the index
	Outputs:
		data_matrix: a matrix containing the flattened image vectors of dimension (number_pics x 1025)
		possible: an updated vector of the input "possible" vector containing the indicies that were not accessed
	'''
	data_matrix = np.zeros(shape=(number_pics,1025)) #predefine matrix that will hold flattened image vectors
	i = 0 #counter for number of pictures that have been accessed
	while i<number_pics: #loops until the target number of pictures have been accessed
		if (possible == []): #this means there are no more pictures to access, an error has likely occured
			print "error getting pictures"
			print name
			return
		index = choice(possible) #random function that chooses an index from possible indicies
		possible.remove(index) #removes this chosen index from indicies of possible images that havent been chosen before
		filename = name+str(index)+'.jpg' #defines the filename to be searched using the chosen index
		try: #try to access the filename, which may or may not exist (not all individuals have the same numebr of pictures)
			im = imread("cropped/"+filename) #read file from folder named "cropped" as a 32x32 matrix of numbers
			im = np.array(im, dtype = "f") #convert numbers to floats
			im = im/255 #normalize each pixel to a number between 0 and 1
			im = np.asarray(im).reshape(-1) #flatten matrix to a shape of 1x1024
			im = np.append([1.0], im) #add x0 = 1 at beginning for theta_0, the constant shift in the linear function
			im = im.reshape(1,1025) #define shape of vector, the shape of "im" is now [[1x1025]]
			im = im[0] #store vector with single brackets such that "im" is now [1x1025]
			data_matrix[i] = im #update the data matrix with the flattened image
			i+=1 #update the number of images that have been accessed
		except: continue #unsuccessful at accessing the filename, continue the loop searching for a different index
	return data_matrix, possible #return outputs

def get_cropped_data(actor_names, training_size, test_size, validation_size, total_range):
	'''
	Function: acquires matrix of flattened image vectors of individuals from image files stored on the computer
	Input:
		actor_names: a list of strings of the actor/actress names, also the file names of the images
		training_size: number of images required for the training set of images (int)
		test_size: number of images required for the test set of images (int)
		validation_size: number of images required for the validation set of images (int)
		total_range: the range of indicies that image files are numbered off (int)
	Output:
		training_set: list where each element is a training_size by 1025 matrix of flattened vector images for a single individual
		test_set: list where each element is a test_size by 1025 matrix of flattened vector images for a single individual
		validation_set: list where each element is a validation_size by 1025 matrix of flattened vector images for a single individual
	'''
	training_set = [] #define training_set as list
	test_set = [] #define test_set as list
	validation_set = [] #define validation_set as list
	for name in range(len(actor_names)): #loops through each actor/actress in the list of actor/actress names
		possible = range(total_range) #define a vector of possible indicies from the total range of indicies in the input
		data_matrix, possible = get_pics(training_size, actor_names[name], possible) #call function to return a matrix training_size number images
		training_set.append(data_matrix) #append matrix to training set list
		data_matrix, possible = get_pics(test_size, actor_names[name], possible) #call function to return a matrix test_size number images
		test_set.append(data_matrix)#append matrix to test set list
		data_matrix, possible = get_pics(validation_size, actor_names[name], possible) #call function to return a matrix validation_size number images
		validation_set.append(data_matrix)#appened matrix to validation set list
	return training_set, test_set, validation_set #return training, test, and validation set of flattened vector images of all actors specified

###################################################################################################
#FUNCTIONS FOR GRADIENT DESCENT 
###################################################################################################

def get_error(theta, data, desired_result):
	'''
	Function: computes error of classifier function from a given desired result
	Input: 
		theta: classifier function (matrix or vector)
		data: image data used to calculate output of classifier function (matrix)
		desired_result: desired result from classifier function (matrix or vector)
	output:
		error: the calculated difference between the classifier function output and the desired result (matrix of vector)
	'''
	h_theta = np.dot(data.T, theta.T) #dot product to calculate output of classifier function
	error = h_theta-desired_result #subtracting desired result from classifier function for error
	return error #return error

def get_grad_vector(theta, data, desired_result):
	'''
	Function: returns a gradient vector or matrix of the direction to minimize cost function
	Input: 
		theta: classifier function (matrix or vector)
		data: image data used to calculate output of classifier function (matrix)
		desired_result: desired result from classifier function (matrix or vector)
	Output:
		grad_vector: a gradient vector or matrix of the direction to minimize cost function
	'''
	error = get_error(theta, data, desired_result) #acquire error between classifier function and desired result
	grad_vector = np.dot(data, error) #dot product of image data with error for gradient vector of cost function
	return grad_vector #return gradient vector

def get_finite_diff_grad_vector(theta, data, desired_result, step_size):
	'''
	Function: eturns a gradient vector or matrix of the direction to minimize cost function
	Input: 
		theta: classifier function (matrix or vector)
		data: image data used to calculate output of classifier function (matrix)
		desired_result: desired result from classifier function (matrix or vector)
		step_size: step sized used in approximation of gradient (float)
	Output:
		grad_vector: a gradient vector or matrix of the direction to minimize cost function
	'''
	grad_vector = np.ones(theta.shape) #intiialize gradient vector/matrix
	for i in range(theta.shape[0]):
		for j in range(theta.shape[1]):#iterate through all elements of theta
			theta_stepped = theta.copy() #copy theta
			theta_stepped[i][j] = theta[i][j] + step_size #forward perterb theta by step size
			fwd = cost_function(theta_stepped, data, desired_result)*data.shape[1]*2 #calculate forward perturbed cost
			theta_stepped[i][j] = theta[i][j] - 2*step_size#backward perterb theta by step size
			bwd = cost_function(theta_stepped, data, desired_result)*data.shape[1]*2 #calculate backward perturbed cost
			grad_vector[i][j] = (fwd-bwd)/(2*step_size) #approximate gradient
	return grad_vector.T #return gradient vector

def unit_vector(vector):
	''' 
    Function: Returns the unit vector of the vector
    '''
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	'''
	Function: compresses v1 and v2 into vectors and returns the angle between the vectors
	Input: 
		v1: first vector or matrix
		v2: second vector or matrix
	Output: 
		angle between v1 and v2 in radians
	'''
	v1 = v1.reshape(1, np.size(v1)) #reshape into vector
	v1 = v1[0]
	v2 = v2.reshape(1, np.size(v2)) #reshape into vector
	v2 = v2[0]
	v1_u = unit_vector(v1) #calculate unit vector direction
	v2_u = unit_vector(v2) #calculate unit vector direction
	angle = np.arccos(np.clip(np.dot(v1_u.T, v2_u), -1.0, 1.0))
	if angle > math.pi: angle = angle-math.pi
	return angle

def cost_function(theta, data, desired_result):
	'''
	Function: returns the cost function of a given theta, x, and target output y using squared error
	Input:
		theta: classifier function (matrix or vector)
		data: image data used to calculate output of classifier function (matrix)
		desired_result: desired result from classifier function (matrix or vector)
	Output: 
		cost: the average square error of the classifer function output compared to the target output (float)
	'''
	error = get_error(theta, data, desired_result) #aquire error between classifier output and target output
	error_squared = np.square(error) #square errors 
	cost = np.average(error_squared) #average all errors
	return cost #return cost 

def grad_descent(theta, training_data_x, training_data_y, alpha, target_cost, print_data):
	'''
	Function: runs a gradient descent to minimize the cost function to a target cost by updating the theta classifier matrix 
	Input:
		theta: intial guess of classifier function (matrix or vector)
		training_data_x: image data used to calculate output of classifier function (matrix)
		training_data_y: target outputs also used to calculate output of classifier function (matrix)
		alpha: used to reduce the step size of each iteration such that the gradient descent does not diverge (float)
		target_cost: the target cost of the gradient descent (float)
		print_data: whether to print some statistics from the gradient descent (final cost, # iterations, alpha used for descent)
	Output: 
		theta_new: the theta classifier matrix which satisfies the target cost for the training set data (matrix)
		theta_history: a list of the thetas at every 10th iteration of the gradient descent
	'''
	theta_old = theta #store initially guessed theta as the old theta
	act_cost = cost_function(theta_old, training_data_x, training_data_y) #calculate the cost of the guessed theta
	theta_history = [] #intilize list to contain thetas every 10th iteration
	i = 0 #initialize iteration counter
	while act_cost>target_cost: #continue to gradient descent until the cost is below the target cost is reached
		if i == 0: #check if it is the first time in the loop
			act_cost_zero = act_cost #store cost of guessed theta
			theta_zero = theta_old #store guessed theta
		if i%10 == 0: #check if 10th iteration
			theta_history.append(theta_old) #store theta in history
		if i > 1000000: #check if a million iterations have been reached, to make sure the loop doesnt run indefinately
			print "1000000 iterations, timeout" #display that a million iterations have been run
			break #break loop 
		grad_vector = get_grad_vector(theta_old, training_data_x, training_data_y) #obtain gradient vector of cost function
		theta_new = theta_old - alpha*grad_vector.T #update theta using gradient descent
		new_act_cost = cost_function(theta_new, training_data_x, training_data_y) #calculate new cost with updated theta
		if act_cost_zero < new_act_cost or math.isnan(new_act_cost): #redefine alpha if newly calculated cost diverges from cost of initially guessed theta
			theta_old = theta_zero #reinstate initially guessed theta
			act_cost = act_cost_zero #reinstate cost of initially guessed theta
			alpha = alpha*0.95 #reduce alpha by a factor of 0.95
			theta_history = [] #clear theta history 
			i == 0 #restart iteration count
		else: #cost did not diverge from cost of previous iteration
			theta_old = theta_new #store updated theta as the old theta
			act_cost = new_act_cost #update cost of updated theta as the cost to be compared against next iteration
			i+=1 #update number of iterations
		if alpha == 0.0: #check if alpha has been reduced so much that it is zero
			print "Error: alpha reduced to zero" #print error message
			return theta_new, theta_history #break loop and exit function
	if print_data: #check if the data is to be displayed
		print "Final Cost:", cost_function(theta_new, training_data_x, training_data_y)
		print "Total Number of Iterations:", i
		print "Actual alpha Used:", alpha
	return theta_new, theta_history #return final theta and history of thetas

###################################################################################################
#FUNCTIONS FOR ACQUIRING DESIRED OUTPUTS
###################################################################################################

def get_labels_binary(number_pics):
	'''
	Function: generates a list of vectors corresponding to the desired output of the classifier function
	Input: 
		number_pics: the required size of the desired output vector, corresponding to the number of pictures in the set (int)
	Output:
		binary_labels: a list of two vectors, the first is a vector of all -1.0 and the second a vector of all 1.0
	'''
	binary_labels = [np.ones((number_pics, 1))*-1.0, np.ones((number_pics, 1))*1.0] #using numpy to generate vectors and scaling appropriately
	return binary_labels #return list

def get_labels_matrix(actor_names, number_pics):
	'''
	create matrix of target outputs for multiple actor/actress classification
	'''
	for i in range(len(actor_names)):
		actor_label = np.zeros((6,number_pics))
		actor_label[i] = np.ones((1,number_pics))
		if i == 0: label_matrix = actor_label.copy()
		else: label_matrix = np.concatenate((label_matrix, actor_label), axis=1)
	return label_matrix

###################################################################################################
#FUNCTIONS FOR TESTING PERFORMANCE
###################################################################################################

def test_data_binary(theta, data_set):
	'''
	Function: computes the number of correctly classified images by the hypothesis function for binary classification
	Inputs:
		theta: hypothesis matrix, calculated through the gradient descent function
		data_set: matrix of flattened image vectors, in which the first half of the vectors are classified with "-1" and the second half classified with "+1"
	Outputs:
		correct: number of images correctly classified
		incorrect: number of images incorrectly classified 
	'''
	correct = 0 #initialize counter for correctly classified images
	incorrect = 0 #initialize counter for incorrectly classified images
	h_theta = np.dot(theta, data_set) #compute output vector of hypothesis function
	for i in range(h_theta.shape[1]): #loop through all indicies of output vector
		if int(i/(0.5*h_theta.shape[1])) == 0: #check whether this is an output from the first half of vectors (classified with "-1")
			if h_theta[0][i]<0: correct+=1 #output is less than zero and correctly classified
			else: incorrect+=1 #output is greater than zero and incorrectly classified
		else:  #output from the second half of vectors (classified with "+1")
			if h_theta[0][i]>0: correct+=1 #output is greater than zero and correctly classified
			else: incorrect+=1 #output is less than zero and incorrectly classified
	return correct, incorrect #return final counters of correct and incorrect

def test_data_matrix(theta, data_set, size):
	'''
	Function: determine number of correctly/incorrectly classified 
	Input:
		theta: hypothesis matrix, calculated through the gradient descent function
		data_set: matrix of flattened image vectors
		size: number of images for each actor/actress
	Output:
		correct: number of images correctly classified
		incorrect: number of images incorrectly classified 
	'''
	correct = 0 #initialize counter for correctly classified images
	incorrect = 0 #initialize counter for incorrectly classified images
	h_theta = np.dot(theta, data_set) #compute output vector of hypothesis function
	for i in range(h_theta.shape[1]): #loop through all images
		if int(i/size) ==  h_theta[:,i].argmax(): correct +=1 #check if index with highest output is the correct classification, if correct, update correct counter
		else: incorrect+=1	#if incorrect, update incorrect counter
	return correct, incorrect #return final counters of correct and incorrect

def concatenate_matrix(matrix):
	'''
	combine a list of multiple matricies into one large matrix
	'''
	for i in range(len(matrix)):
		if i == 0: new_matrix = matrix[i]
		else: new_matrix = np.concatenate((new_matrix, matrix[i]), axis=0)
	return new_matrix

###################################################################################################
#FUNCTIONS FOR PLOTTING
###################################################################################################

def get_graph_cost_iteration(theta_history, training_data_x, training_data_y, validation_data_x, validation_data_y):
	'''
	plots the cost function of training and validiation sets with respect to iterations of gradient descent
	'''
	cost_training = []
	cost_validation = []
	x = arange(len(theta_history))*10
	for i in range(len(theta_history)):
		cost_training.append(cost_function(theta_history[i], training_data_x, training_data_y))
		cost_validation.append(cost_function(theta_history[i], validation_data_x, validation_data_y))
	try:
		plt.yscale('log')
		plt.suptitle('Cost Training vs Validation Sets')
		plt.xlabel('Iterations')
		plt.ylabel('Cost')
		plt.plot(x, cost_training, '-b', label="Training")
		plt.plot(x, cost_validation, '-r', label="Validation")
		plt.legend(loc='best')
		plt.show()
	except:
		print "Failed to Plot Cost Training vs Validation Sets"
		return
	return

def get_graph_performance(theta_history, training_data_x, training_size, validation_data_x, validation_size):
	'''
	plots percent correctly classified of training and validiation with respect to iterations of gradient descent for binary classification
	'''
	performance_training = []
	performance_validation = []
	x = arange(len(theta_history))*10
	for i in range(len(theta_history)):
		correct, incorrect = test_data_binary(theta_history[i], training_data_x)
		performance_training.append((float(correct)*100/float(correct+incorrect)))
		correct, incorrect = test_data_binary(theta_history[i], validation_data_x)
		performance_validation.append((float(correct)*100/float(correct+incorrect)))
	try:
		plt.suptitle('Performance Training vs Validation Sets')
		plt.xlabel('Iterations')
		plt.ylabel('Percent Correctly Classified')
		plt.plot(x, performance_training, '-b', label="Training")
		plt.plot(x, performance_validation, '-r', label="Validation")
		plt.legend(loc='best')
		plt.show()
	except:
		print "Failed to Plot Performance Training vs Validation Sets"
		return
	return

def get_graph_performance_matrix(theta_history, training_data_x, training_size, validation_data_x, validation_size):
	'''
	plots percent correctly classified of training and validiation with respect to iterations of gradient descent 
	'''
	performance_training = []
	performance_validation = []
	x = arange(len(theta_history))*10
	for i in range(len(theta_history)):
		correct, incorrect = test_data_matrix(theta_history[i], training_data_x, training_size)
		performance_training.append((float(correct)*100/float(correct+incorrect)))
		correct, incorrect = test_data_matrix(theta_history[i], validation_data_x, validation_size)
		performance_validation.append((float(correct)*100/float(correct+incorrect)))
	try:
		plt.suptitle('Performance Training vs Validation Sets')
		plt.xlabel('Iterations')
		plt.ylabel('Percent Correctly Classified')
		plt.plot(x, performance_training, '-b', label="Training")
		plt.plot(x, performance_validation, '-r', label="Validation")
		plt.legend(loc='best')
		plt.show()
	except:
		print "Failed to Plot Performance Training vs Validation Sets"
		return
	return

def plot6d_1(cost, angle):
	'''
	Function: plot angle between gradients vs location on cost function
	'''
	try:
		plt.suptitle('Difference between Finite Diff. Gradient and Actual Gradient vs Location on Cost Function')
		plt.xlabel('J(theta), Cost')
		plt.ylabel('Vector Angle between Flattened Matricies (Radians)')
		plt.xscale('log')
		plt.plot(cost, angle, 'b', label="Step Size of 0.01")
		plt.legend(loc='best')
		plt.show()
	except:
		print "Failed to Plot Difference between Finite Diff. Gradient and Actual Gradient vs Location on Cost Function"
		return
	return

def plot6d_2(step_sizes, angle_same_theta):
	'''
	Function: plot angle between gradients vs step size
	'''
	try:
		plt.suptitle('Difference between Finite Diff. Gradient and Actual Gradient vs Step Size')
		plt.xlabel('Step Size')
		plt.ylabel('Vector Angle between Flattened Matricies (Radians)')
		plt.xscale('log')
		plt.yscale('log')
		plt.plot(step_sizes, angle_same_theta, 'b', label="Optimized Theta")
		plt.legend(loc='best')
		plt.show()
	except:
		print "Failed to Plot Difference between Finite Diff. Gradient and Actual Gradient vs Step Size"
		return
	return

def display_theta_images(act, theta): 
	'''
	display image of theta
	'''
	images = []
	for i in range(theta.shape[0]):
		print "Displaying:", act[i]
		theta_image = theta[i][1:]*255
		theta_image = theta_image.reshape(32,32) #reshape vector into an image
		try:
			plt.imshow(theta_image)
			plt.show()
			plt.imshow(theta_image)
		except:
			print "Failed to display", act[i], "theta"
	return images

###################################################################################################
#FUNCTIONS FOR PRINTING OUTPUTS
###################################################################################################

def print_test_results(correct, incorrect):
	'''
	Function: print test results
	'''
	print "correct:", correct
	print "incorrect:", incorrect
	print "Percent Correct:", int(float(correct)*100/float(correct+incorrect)), "%"

def print_double_line():
	print "========================================="
	return

def print_single_line():
	print "-----------------------------------------"
	return

###################################################################################################
#FUNCTIONS FOR ASSIGNMENT OUTPUTS
###################################################################################################

def part3():
	print_double_line()
	print "Part 3: Binary Gradient Descent"
	print_single_line()
	target_cost = 0.05
	alpha = 1
	training_size = 100
	test_size = 0
	validation_size = 10
	act = ["Baldwin", "Carell"]
	training_data_x, not_needed, validation_data_x = get_cropped_data(act, training_size, test_size, validation_size, 200) 
	training_data_y = get_labels_binary(training_size) 
	training_data_x = concatenate_matrix(training_data_x).T
	training_data_y = concatenate_matrix(training_data_y)
	validation_data_y = get_labels_binary(validation_size)
	validation_data_x = concatenate_matrix(validation_data_x).T
	validation_data_y = concatenate_matrix(validation_data_y)
	validation_data_y = concatenate_matrix(validation_data_y).T
	theta = np.ones((1,1025)) #checked
	theta_optimized, theta_history =  grad_descent(theta, training_data_x, training_data_y, alpha, target_cost, True)
	get_graph_cost_iteration(theta_history, training_data_x, training_data_y, validation_data_x, validation_data_y)
	get_graph_performance(theta_history, training_data_x, training_size, validation_data_x, validation_size)
	print_single_line()
	print "Training set"
	correct, incorrect = test_data_binary(theta_optimized, training_data_x)
	print_test_results(correct, incorrect)
	print_single_line()
	print "Validation set"
	correct, incorrect = test_data_binary(theta_optimized, validation_data_x)
	print_test_results(correct, incorrect)
	return

def part4():
	print_double_line()
	print "Part 4: Visualizing with Training set of 2 and 100 images"
	print_single_line()
	target_cost = 0.005
	training_size2 = 2
	training_size100 = 100
	test_size = 0
	validation_size = 0
	act = ["Baldwin", "Carell"]
	training_data_x2, test_data_x, validation_data_x = get_cropped_data(act, training_size2, test_size, validation_size, 200) 
	training_data_x100, test_data_x, validation_data_x = get_cropped_data(act, training_size100, test_size, validation_size, 200) 
	training_data_y2 = get_labels_binary(training_size2) 
	training_data_x2 = concatenate_matrix(training_data_x2).T
	training_data_y2 = concatenate_matrix(training_data_y2)
	training_data_y100 = get_labels_binary(training_size100) 
	training_data_x100 = concatenate_matrix(training_data_x100).T
	training_data_y100 = concatenate_matrix(training_data_y100)
	theta = np.ones((1,1025)) 
	print "2 Images"
	alpha = 1
	theta_optimized2, theta_history =  grad_descent(theta, training_data_x2, training_data_y2, alpha, target_cost, True)
	theta = np.ones((1,1025)) 
	print_single_line()
	print "100 Images"
	alpha = 1
	theta_optimized100, theta_history =  grad_descent(theta, training_data_x100, training_data_y100, alpha, target_cost, True)
	theta_image = theta_optimized2[0][1:]
	theta_image = theta_image.reshape(32,32)
	plt.imshow(theta_image)
	plt.show()
	theta_image = theta_optimized100[0][1:]
	theta_image = theta_image.reshape(32,32)
	plt.imshow(theta_image)
	plt.show()
	print_double_line()
	return

def part5():
	print_double_line()
	print "Part 5: Effect of Training Set Size on Gender Classification Performance"
	print_single_line()
	alpha = 1
	target_cost = 0.15
	training_size = 1
	test_size = 10
	validation_size = 10
	iterations_per_size = 10
	final_training_size = 140
	classification_set = ["women", "men"]
	act =['Drescher', 'Ferrera', 'Chenoweth', 'Baldwin', 'Hader', 'Carell']
	act_test = ['Bracco', 'Gilpin', 'Harmon', 'Butler', 'Radcliffe', 'Vartan']
	correct_vector = np.ones((4, final_training_size))
	for i in range(final_training_size):
		correct_vector[0][i] = training_size
		correct_training = np.ones((1,iterations_per_size))
		correct_validation = np.ones((1,iterations_per_size))
		correct_testing = np.ones((1,iterations_per_size))
		for j in range(iterations_per_size):
			training_data_x, not_needed1, validation_data_x = get_cropped_data(act, training_size, 0, validation_size, 200) 
			training_data_x = concatenate_matrix(training_data_x).T
			training_data_y = get_labels_binary(training_size*3)
			training_data_y = concatenate_matrix(training_data_y)	
			not_needed3, test_data_x, not_needed5 = get_cropped_data(act_test, 0, test_size, 0, 200) 
			validation_data_x = concatenate_matrix(validation_data_x).T
			test_data_x = concatenate_matrix(test_data_x).T
			theta = np.ones((1, 1025)) 
			theta_optimized, theta_history =  grad_descent(theta, training_data_x, training_data_y, alpha, target_cost, False)
			correct, incorrect = test_data_binary(theta_optimized, training_data_x)
			correct_training[0][j] = int(float(correct)*100/float(correct+incorrect))
			correct, incorrect = test_data_binary(theta_optimized, validation_data_x)
			correct_validation[0][j] = int(float(correct)*100/float(correct+incorrect))
			correct, incorrect = test_data_binary(theta_optimized, test_data_x)
			correct_testing[0][j] = int(float(correct)*100/float(correct+incorrect))
			j +=1
		correct_vector[1][i] = np.average(correct_training[0])
		correct_vector[2][i] = np.average(correct_validation[0])
		correct_vector[3][i] = np.average(correct_testing[0])
		print "Finished Training set of size:", training_size
		training_size+=1
		i+=1
	try:
		plt.suptitle('Effect of Training Set Size on Gender Classification Performance')
		plt.xlabel('Number of Images in the Training Set (Per actor/actress)')
		plt.ylabel('Percent Correctly Classified')
		plt.plot(correct_vector[0], correct_vector[1], '-b', label="Training Set")
		plt.plot(correct_vector[0], correct_vector[2], '-r', label="Validation Set")
		plt.plot(correct_vector[0], correct_vector[3], '-y', label="Test Set of Different Actors/Actresses")
		plt.legend(loc='best')
		plt.show()
	except:
		print "Failed to Plot Effect of Training Set Size on Gender Classification Performance"
		return
	return

def part6d():
	print_double_line()
	print "Part 6d: Comparing Finite Differences Gradient"
	print_single_line()
	training_size = 100
	test_size = 10
	validation_size = 10
	act =['Drescher', 'Ferrera', 'Chenoweth', 'Baldwin', 'Hader', 'Carell']
	training_data_x, test_data_x, validation_data_x = get_cropped_data(act, training_size, test_size, validation_size, 200) 
	training_data_y = get_labels_matrix(act,training_size).T
	training_data_x = concatenate_matrix(training_data_x).T
	theta = np.ones((6, 1025))
	alpha = 1
	target_cost = 10
	theta_optimized, theta_history =  grad_descent(theta, training_data_x, training_data_y, alpha, target_cost, True)
	angle = np.ones((len(theta_history), 1))
	cost = np.ones((len(theta_history), 1))
	for i in range(len(theta_history)):
		finite_diff_grad_vector = get_finite_diff_grad_vector(theta_history[i], training_data_x, training_data_y, 0.01)
		grad_vector = get_grad_vector(theta_history[i], training_data_x, training_data_y)
		angle[i][0] = angle_between(finite_diff_grad_vector, grad_vector)
		cost[i][0] = cost_function(theta_history[i], training_data_x, training_data_y)
	plot6d_1(cost, angle)
	
	num_step_sizes = 5
	angle_same_theta = ones(num_step_sizes)
	step_sizes = ones(num_step_sizes)
	for i in range(len(step_sizes)):
		step_sizes[i] = step_sizes[i]*(0.1**i)
	grad_vector = get_grad_vector(theta_optimized, training_data_x, training_data_y)
	for i in range(num_step_sizes):
		finite_diff_grad_vector = get_finite_diff_grad_vector(theta_optimized, training_data_x, training_data_y, step_sizes[i])
		angle_same_theta[i] = angle_between(finite_diff_grad_vector, grad_vector)
	plot6d_2(step_sizes, angle_same_theta)
	return

def part7():
	print_double_line()
	print "Part 7: Performance of Training and Validaiton Sets"
	print_single_line()
	alpha = 1
	target_cost = 0.02
	training_size = 100
	test_size = 10
	validation_size = 10
	act =['Drescher', 'Ferrera', 'Chenoweth', 'Baldwin', 'Hader', 'Carell']
	training_data_x, test_data_x, validation_data_x = get_cropped_data(act, training_size, test_size, validation_size, 200) 
	training_data_y = get_labels_matrix(act,training_size).T
	training_data_x = concatenate_matrix(training_data_x).T
	test_data_x = concatenate_matrix(test_data_x).T
	validation_data_x = concatenate_matrix(validation_data_x).T
	validation_data_y = get_labels_matrix(act,validation_size).T
	theta = np.ones((6, 1025))
	theta_optimized, theta_history =  grad_descent(theta, training_data_x, training_data_y, alpha, target_cost, True)
	get_graph_cost_iteration(theta_history, training_data_x, training_data_y, validation_data_x, validation_data_y)
	get_graph_performance_matrix(theta_history, training_data_x, training_size, validation_data_x, validation_size)
	return

def part8():
	print_double_line()
	print "Part 8: Visualize theta images"
	print_single_line()
	alpha = 1
	target_cost = 0.25
	training_size = 100
	test_size = 0
	validation_size = 0
	act =['Drescher', 'Ferrera', 'Chenoweth', 'Baldwin', 'Hader', 'Carell']
	training_data_x, not_neede1, not_needed2 = get_cropped_data(act, training_size, test_size, validation_size, 200) 
	training_data_y = get_labels_matrix(act,training_size).T
	training_data_x = concatenate_matrix(training_data_x).T
	theta = np.ones((6, 1025))
	theta_optimized, theta_history =  grad_descent(theta, training_data_x, training_data_y, alpha, target_cost, True)
	display_theta_images(act, theta_optimized)
	print_double_line()



if __name__ == "__main__":
###################################################################################################
#UNCOMMENT FUNCTIONS FOR OUTPUT OF CORRESPONDING PART IN ASSIGNMENT 
###################################################################################################
	part3()
	part4()
	part5()
	part6d()
	part7()
	part8()





      


