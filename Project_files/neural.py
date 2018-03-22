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

	set_inputs_binary_pair()



def set_inputs_binary_pair():
	global net, numInputs

	if numInputs == 2:
		net.node[0]['output'] = math.ceil(random.random()*2 - 1)
		net.node[1]['output'] = math.ceil(random.random()*2 - 1)
	else:
		print "not valid"

def run_epoch():
	global net, idealOutput, totalNodes, numOutputs
	for i in range(0,2):
		for j in range(0,2):
			net.node[0]['output'] = i
			net.node[1]['output'] = j
			set_ideal_output()
			forward_prop()
			back_prop()	
			#print idealOutput


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

	#set_ideal_output

	for i in reversed(range(numInputs, len(net.nodes()))):
		calc_error(i)

	for i in range(numInputs, len(net.nodes())):
		update_weights(i)


def calc_error(node):
	global net, idealOutput, outputNodes

	net.node[node]['error'] = 0

	if(net.node[node]['type'] == "output"):
		net.node[node]['error'] = -(idealOutput[node] - net.node[node]['output'])
		net.node[node]['delta'] = net.node[node]['error'] * calc_deriv(net.node[node]['output'])

	elif(net.node[node]['type'] == "hidden"):
		for outNode, edge in net.adj[node].iteritems():
			if(outNode > node):
				net.node[node]['error'] += edge['weight'] * net.node[outNode]['delta']
		net.node[node]['delta'] = net.node[node]['error'] * calc_deriv(net.node[node]['output'])

def calc_deriv(value):
	return value * (1.0 - value)

def update_weights(node):
	global net, learning_rate
	for inNode, edge in net.adj[node].iteritems():
		if(inNode < node):
			edge['weight'] = edge['weight'] - learning_rate * net.node[node]['delta'] * net.node[inNode]['output']

def set_ideal_output():
	global net, idealOutput, totalNodes, numOutputs

	if net.node[0]['output'] == 1 and net.node[1]['output'] == 1:
		idealOutput[totalNodes - numOutputs] = 1
	else:
		idealOutput[totalNodes - numOutputs] = 0
	#print idealOutput
def test_epoch():
	global net, idealOutput, totalNodes, numOutputs
	for i in range(0,2):
		for j in range(0,2):
			net.node[0]['output'] = i
			net.node[1]['output'] = j
			set_ideal_output()
			forward_prop()
			print("cycle: {}".format(i))
			print("input: {} {}".format(net.node[0]['output'], net.node[1]['output']))
			print("expected: {}".format(idealOutput[totalNodes - numOutputs]))
			#print idealOutput
			print("actual: {}".format(net.node[totalNodes - 1]['output']))	



def draw_network(g, layers, inLay, numInputs, numOutputs):
	print "drawing network"
	plt.subplot(121)
	layout = grid_layout(g, layers, inLay, numInputs, numOutputs)

	nx.draw(g, pos=layout, edge_cmap='red', linewidths=6, with_labels=True)

	plt.show()



num_layers = 2
num_in_layer = 2
numInputs = 2
numOutputs = 1
totalNodes = numInputs + numOutputs + num_layers * num_in_layer
outputNodes = list(range(totalNodes - numOutputs, totalNodes))
idealOutput = {}
learning_rate = 1
numEpochs = 25000

net = nx.Graph()
init_network(num_layers, num_in_layer, numInputs, numOutputs)





#print net.node[totalNodes - 1]['output']

print "pre-training predictions"

test_epoch()

print "training"


for i in range(1, numEpochs):

	run_epoch()

	if i % (numEpochs / 10) == 0:
		print("{}% complete".format(1.0 * i / numEpochs * 100.0))


	'''
	if i % 1000 == 0:
		
		print("cycle: {}".format(i))
		print("input: {} {}".format(net.node[0]['output'], net.node[1]['output']))
		print("expected: {}".format(idealOutput[totalNodes - numOutputs]))
		#print idealOutput
		print("actual: {}".format(net.node[totalNodes - 1]['output']))
		#print("expected: {}\nactual:{}\n".format(idealOutput[totalNodes - numOutputs], net.node[totalNodes - 1]['output']))
	'''


test_epoch()

#for i in range(0, len(net.nodes())):
#	print net.node[i]



#draw_network(net, num_layers, num_in_layer, numInputs, numOutputs)
