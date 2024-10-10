# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 00:12:04 2024

@author: Bob
"""
import numpy as np

# Static Variables
N = 2 #Number of inputs
NUM_HIDDEN_LAYERS = 2 # Number of hidden layers
M = [2, 2] # Number of nodes in each hidden layer
NUM_NODES_OUTPUT = 1 #Number of nodes in the output layer

num_nodes_previous = N # number of nodes in the previous layer

network = {}

for layer in range(NUM_HIDDEN_LAYERS + 1):
    
    # determine name of the layer
    if layer == NUM_HIDDEN_LAYERS:
        layer_name = 'output'
        num_nodes = NUM_NODES_OUTPUT
    else:
        layer_name = f"layer_{layer+1}"
        num_nodes = M[layer]
        
    # Initialize weights and biases associated with each node in the current layer
    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = f"node_{node+1}"
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias':np.around(np.random.uniform(size=1), decimals=2)
            }
        
    num_nodes_previous = num_nodes
    
print(network)

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {}
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes
        
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

from random import seed

np.random.seed(42)

inputs = np.around(np.random.uniform(size=5), decimals=2)

def node_activation(weighted_sum):
    return 1.0/ (1.0 + np.exp(-1 * weighted_sum))

def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions
