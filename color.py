from PIL import Image
# matplotlib only supports PNG images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from math import exp
from random import seed
from random import random
from glob import glob
ImageList=[]
GrayImage=[]
DataList=[]
Array=np.array([[4.4679,-3.5873,0.1193],[-1.2186,2.3809,-0.1624],[0.0497,-0.2439,1.2045]])

def color_to_grey(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])
#using Luminosity formula for convertion of RGB into GrayScale
    R = (R *.21)
    G = (G *.72)
    B = (B *.07)
        
    Sum = (R+G+B)
        
        
#changing the RGB values of the image to the newly calculated Greyscale values

    for i in range(3):
        grayImage[:,:,i] = Sum
 
    return grayImage       


#This network should have 10 inputs:9 pixels of 3*3 grid, the standrad derivationn of cental pixel and the bias
# Three hidden neurals and three output values RGB
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    for neuron in hidden_layer:
        temp=neuron['weights']
        temp[4]=1.0
        neuron['weights']=temp
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    
    return network


 
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    #print("The num of weights is "+str(len(weights)))
    #print("The input is "+str(len(inputs)))
    #print(inputs[0])
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation
 
# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, grid):
    #print("The input is "+str(len(grid)))
    
    inputs = grid
    for layer in network:
        new_inputs = []
        #i=0
        for neuron in layer:
            #i=i+1
            #print("The num of weights is "+str(len(neuron['weights'])))
            #print(i)
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
            #
def train_network(network,grayimg,img,l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        shape=grayimg.shape
        for i in range(1,shape[0]-1):
            for j in range(1,shape[1]-1):
                grid=np.array(grayimg[i-1:i+2,j-1:j+2, 0])
                grid=grid.reshape(1,9)
                #print(grid[0])
                outputs = forward_propagate(network, grid[0])
                #print("Enter the expect")
                #expected = [0 for i in range(n_outputs)]
                #expected[row[-1]] = 1
                expected=np.array(img[i,j,:])
                #print(expected)
                
                #expected=list(expected)
                outputs=np.array(outputs)
                #outputs=outputs.reshape(3,1)
                #outputs=Array*outputs
                sum_error +=  sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                backward_propagate_error(network, expected)
                update_weights(network, grid[0], l_rate)
        print('>epoch=%d, lrate=%.3f' % (epoch, l_rate)+' sum of error: '+str(sum_error))
 
# Prediction with NN
def predict(network, grid):
	outputs = forward_propagate(network, grid)
	return outputs






#image1=Image.open("/Users/shiyunhao/Desktop/colorization-master/bird2.jpg")
#image1.show()
#image1.save('/Users/shiyunhao/Desktop/colorization-master/bird2.png')
'''
img1=mpimg.imread('/Users/tanuj180317/Desktop/images/bird.png')
grayImage1 = color_to_grey(img1) 
plt.imshow(img1)
plt.show()
plt.imshow(grayImage1)
plt.show()
ImageList.append(img1)
GrayImage.append(grayImage1)

img2=mpimg.imread('/Users/tanuj180317/Desktop/images/bird2.png')
grayImage2 = color_to_grey(img2) 
plt.imshow(grayImage2)
plt.show()
ImageList.append(img2)
GrayImage.append(grayImage2)


seed(1)
network = initialize_network(9, 3, 3)

img_mask = '/Users/Amanda/Documents/Rutgers/f18/520/colorizer/*.png'
img_names = glob(img_mask)
for fn in img_names:
    print('processing %s...' % fn,)
    img = mpimg.imread(fn, 0)
    grayImage = color_to_grey(img)
    plt.imshow(grayImage)
    plt.show()
    ImageList.append(img)
    GrayImage.append(grayImage)
    train_network(network,grayImage,img, 0.5, 30, 3)
    
'''
#Initialize the NN
#seed(1)
#network = initialize_network(9, 3, 3)

#plt.imshow(img2)
#plt.show()
#train_network(network,grayImage2,img2, 0.5, 900, 3)
#for layer in network:
	#print(layer)

seed(1)
#the number of nodes in the NN can be changedhere
network = initialize_network(9, 9, 3)

#train bird1
img = mpimg.imread('/Users/Amanda/Documents/Rutgers/f18/520/colorizer/bird1.png')
grayImage = color_to_grey(img)
plt.imshow(grayImage)
plt.show()
ImageList.append(img)
GrayImage.append(grayImage)
#The learning rate can be changed here, though keeping it low should work.
#Not sure we should increase the number of epochs due to run time
#but it can also be changed here.
train_network(network,grayImage,img, 0.3, 30, 3)

#sanity check: predict bird1    
Tshape=grayImage.shape
Tcolor=np.zeros(img.shape)
for i in range(1,Tshape[0]-1):
    for j in range(1,Tshape[1]-1):
        grid=np.array(grayImage[i-1:i+2,j-1:j+2, 0])
        grid=grid.reshape(1,9)
        prediction = predict(network, grid[0])
        Tcolor[i,j,0] = prediction[0]
        Tcolor[i,j,1] = prediction[1]
        Tcolor[i,j,2] = prediction[2]
plt.imshow(Tcolor)
plt.show()


