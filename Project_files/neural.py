# Imports
import networkx as nx
import numpy
import matplotlib.pyplot as plt
import random
import math

def grid_layout(g, layers, inLay, numInputs, numOutputs):

	layout = {}

	#layout inputs
	for node in range(0,numInputs):
		layout[node] =([0, node]) 

	#layout hidden
	for layer in range(0,layers):
		for node in range(0,inLay):
			layout[numInputs + layer*inLay + node] =([1 + layer, node]) 
	

	#layout outputs
	for node in range(0, numOutputs):
		
		layout[node + numInputs + layers*inLay] =([layers + 1, node]) 

	return layout

def init_network(num_layers, num_in_layer, numInputs, num_outputs):
	global net

	print "initializing network"

	#net.add_nodes_from(range(0,num_layers*num_in_layer + numInputs+numOutputs), summed_input=0, output=0, error=0)

	net.add_nodes_from(range(0,numInputs), output=0, type="input")
	net.add_nodes_from(range(numInputs, num_layers*num_in_layer + numInputs), summed_input=0, output=0, error=0,  delta=0, type="hidden",)
	net.add_nodes_from(range(num_layers*num_in_layer + numInputs, num_layers*num_in_layer + numInputs+numOutputs), summed_input=0, output=0, error=0, delta=0, type="output")

	#construct hidden layers
	for layer in range(1,num_layers):
		for receiving in range(0,num_in_layer):
			for sending in range(0,num_in_layer):
				net.add_edge(numInputs + num_in_layer*layer + receiving, numInputs + num_in_layer *(layer-1) + sending, weight = random.random()*2 - 1)

	#connect to inputs
	for hidden in range(0,num_in_layer):
		for inNode in range(0,numInputs):
			net.add_edge(numInputs + hidden, inNode, weight = random.random()*2 - 1)

	#connect to output
	for hidden in range(0,num_in_layer):
		for outNode in range(0,numOutputs):
			net.add_edge(numInputs +(num_layers - 1)*num_in_layer + hidden,  numInputs + num_layers*num_in_layer + outNode, weight = random.random() * 2 - 1)

	print "initialization complete"

def set_inputs():
	global net, numInputs


def forward_prop():
	global numInputs, net

	for i in range(numInputs, len(net)):
		activate(i)

def activate(node):
	global net

	net.node[node]['summed_input'] = sum_inputs(node)
	net.node[node]['output'] = 1.0/(1 + math.exp(-1 * net.node[node]['summed_input']))
	return node


def sum_inputs(node):
	global net

	sum = 0
	for num, edge in net.adj[node].iteritems():
		
		if(num < node):
			sum += edge['weight'] * net.node[num]['output']
	return sum

def back_prop():
	global net, numInputs
	for i in reversed(range(numInputs, len(net.nodes()))):
		calc_error(i)

	for i in range(numInputs, len(net.nodes()))
		update_weights(i)


def calc_error(node):
	global net, idealOutput

	net.node[node]['error'] = 0

	if(net.node[node]['type'] == "output"):
		net.node[node]['error'] = .5 * pow(idealOutput[node] - net.node[node]['output'], 2)
		net.node[node]['delta'] = net.node[node]['error'] * calc_deriv(net.node[node]['error'])

	elif(net.node[node]['type'] == "hidden"):
		num, edge in net.adj[node].iteritems():
		if(num > node):
			net.node[node]['error'] += edge['weight'] * net.node[num]['delta']
		net.node[node]['delta'] = net.node[node]['error'] * calc_deriv(net.node[node]['error'])

def calc_deric(value):
	return value * (1.0 - value)

def update_weights(node):
	global net, learning_rate
	for num, edge in net.adj[node].iteritems():
		if(num < node):
			edge['weight'] = edge['weight'] + learning_rate * g.node[num]['delta'] * g.node[num]['output']






def draw_network(g, layers, inLay, numInputs, numOutputs):
	print "drawing network"
	plt.subplot(121)
	layout = grid_layout(g, layers, inLay, numInputs, numOutputs)

	nx.draw(g, pos=layout, edge_cmap='red', linewidths=6, with_labels=True)

	plt.show()



num_layers = 3
num_in_layer = 5
numInputs = 5
numOutputs = 2
totalNodes = numInputs + numOutputs + num_layers * num_in_layer

net = nx.Graph()
init_network(num_layers, num_in_layer, numInputs, numOutputs)


forward_prop()

for i in range(0, len(net.nodes())):
	print net.node[i]



#draw_network(net, num_layers, num_in_layer, numInputs, numOutputs)
