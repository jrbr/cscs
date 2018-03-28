# Imports
import networkx as nx
import numpy
import matplotlib.pyplot as plt
import pylab
import random
import math
import graphviz as gv
import pydot
from networkx.drawing.nx_pydot import write_dot
import pandas as pd


########################################################### Network Functions ####################################################################

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


def draw_network(g, layers, inLay, numInputs, numOutputs):
	print "drawing network"
	plt.subplot(121)
	layout = grid_layout(g, layers, inLay, numInputs, numOutputs)

	edge_labels=dict([((u,v,),d['weight'])
             for u,v,d in net.edges(data=True)])

	write_dot(g,"file.gv")
	gv.render('dot', 'png', 'file.gv')
	#gv.view('file.png')

	#nx.draw(g, pos=layout, edge_cmap='red', linewidths=6, with_labels=True)
	#nx.draw_networkx_edges(g, pos=layout, edge_labels = edge_labels)
	#pylab.ylim([-.5,1.5])
	#pylab.xlim([-.5,2.5])

	#pylab.show()

# Assign spreadsheet filename to `file`



########################################################### Learning Base ####################################################################


def set_inputs():
	global net, numInputs

	set_inputs_binary_pair()

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

#Should only be called after calc_error
def get_output_error():
	global net, outputNodes

	sum = 0

	for node in outputNodes:
		sum += math.fabs(net.nodes[node]['error'])

	return sum
################################################### Binary Data ####################################################################

def set_ideal_output():
	return



def set_inputs_binary_pair():
	global net, numInputs

	if numInputs == 2:
		net.node[0]['output'] = math.ceil(random.random()*2 - 1)
		net.node[1]['output'] = math.ceil(random.random()*2 - 1)
	else:
		print "not valid"

def set_AND():
	global net, idealOutput, totalNodes, numOutputs

	if net.node[0]['output'] == 1 and net.node[1]['output'] == 1:
		idealOutput[totalNodes - numOutputs] = 1
	else:
		idealOutput[totalNodes - numOutputs] = 0
	#print idealOutput

def set_OR():
	global net, idealOutput, totalNodes, numOutputs

	if net.node[0]['output'] == 1 or net.node[1]['output'] == 1:
		idealOutput[totalNodes - numOutputs] = 1
	else:
		idealOutput[totalNodes - numOutputs] = 0
	#print idealOutput

def set_NOR():
	global net, idealOutput, totalNodes, numOutputs

	if not (net.node[0]['output'] == 1 or net.node[1]['output'] == 1):
		idealOutput[totalNodes - numOutputs] = 1
	else:
		idealOutput[totalNodes - numOutputs] = 0
	#print idealOutput


def set_XOR():
	global net, idealOutput, totalNodes, numOutputs

	a = net.node[0]['output'] 
	b = net.node[1]['output']
	if a != b and (a == 1 or b == 1):
		idealOutput[totalNodes - numOutputs] = 1
	else:
		idealOutput[totalNodes - numOutputs] = 0
	#print idealOutput

def test_epoch_binary():
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


################################################ Knowledge Level Implementation ########################################################################


def train_epoch_data():
	global net, idealOutput, totalNodes, numOutputs, training_data

	epoch_error = 0
	instances = 0
	for i in training_data.index:
		set_data_in_out(True, i)
		forward_prop()
		back_prop()
		instances += 1
		epoch_error += get_output_error()
	return epoch_error/instances


def test_epoch_data(useTesting, useTraining):
	global net, idealOutput, totalNodes, training_data, testing_data, expected_cat, predicted_cat, outputNodes

	numCorrect = 0
	numTested = 0

	epoch_error = 0

	if useTraining:
		for i in training_data.index:
			set_data_in_out(True, i)
			forward_prop()
			if expected_cat == get_predicted_category():
				numCorrect += 1
			numTested += 1

			for node in outputNodes:
				calc_error(node)
			epoch_error += get_output_error

	if useTesting:
		for i in testing_data.index:
			set_data_in_out(True, i)
			forward_prop()
			if expected_cat == get_predicted_category():
				numCorrect += 1
			numTested += 1

			for node in outputNodes:
				calc_error(node)
			epoch_error += get_output_error


	print("{} out of {} categories predicted correctly: {}%\n Average Error per input:{}".format(numCorrect, numTested, 100.0 * numCorrect / numTested, epoch_error/numTested))
	return epoch_error/numTested

def set_data_in_out(useTraining, i):
	global net, idealOutput, totalNodes, numOutputs, training_data, testing_data, expected_cat

	if useTraining:
		net.node[0]['output'] = training_data['STG'][i]
		net.node[1]['output'] = training_data['SCG'][i]
		net.node[2]['output'] = training_data['STR'][i]
		net.node[3]['output'] = training_data['LPR'][i]
		net.node[4]['output'] = training_data['PEG'][i]
		expected_cat = training_data['UNS'][i]

	else:
		net.node[0]['output'] = testing_data['STG'][i]
		net.node[1]['output'] = testing_data['SCG'][i]
		net.node[2]['output'] = testing_data['STR'][i]
		net.node[3]['output'] = testing_data['LPR'][i]
		net.node[4]['output'] = testing_data['PEG'][i]	
		expected_cat = testing_data['UNS'][i]

	set_data_output(expected_cat)

def set_data_output(cat):
	global idealOutput, outputNodes, correctNode

	#print "setting output"

	#for node in outputNodes:
	#	idealOutput[node] = 0
	idealOutput[correctNode] = 0

	if(cat == "very_low"):
		correctNode = outputNodes[0]
	elif(cat == "Low"):
		correctNode = outputNodes[1]
	elif(cat == "Middle"):
		correctNode = outputNodes[2]
	elif(cat == "High"):
		correctNode = outputNodes[3]
	else:
		print "not a valid category"
	#print idealOutput
	idealOutput[correctNode] = 1
	#print idealOutput


def print_results(verbose):
	global net, idealOutput, totalNodes, outputNodes, expected_cat, predicted_cat

	if not "cat" in net.nodes[outputNodes[0]]:
		net.nodes[outputNodes[0]]['cat'] = "very_low"
		net.nodes[outputNodes[1]]['cat'] = "Low"
		net.nodes[outputNodes[2]]['cat'] = "Middle"
		net.nodes[outputNodes[3]]['cat'] = "High"

	predicted_cat = get_predicted_category()

	

	print ("Expected Category: {}".format(expected_cat))
	print ("Predicted Category: {}".format(predicted_cat))

	if verbose:
		print "Predictive Scores"
		for node in outputNodes:
			print ("{} score: {}".format(net.node[node]['cat'], net.node[node]['output']))

def get_predicted_category():
	global net, idealOutput, totalNodes, outputNodes, expected_cat

	if not "cat" in net.nodes[outputNodes[0]]:
		net.nodes[outputNodes[0]]['cat'] = "very_low"
		net.nodes[outputNodes[1]]['cat'] = "Low"
		net.nodes[outputNodes[2]]['cat'] = "Middle"
		net.nodes[outputNodes[3]]['cat'] = "High"

	

	maxVal = 0
	maxVal_node = outputNodes[0]
	for node in outputNodes:
		if net.nodes[node]['output'] > maxVal:
			maxVal = net.nodes[node]['output']
			maxVal_node = node
	return (net.node[maxVal_node]['cat'])



file = 'user_knowledge.xls'

# Load spreadsheet
print "loading Data..."
training_data = pd.read_excel(file, sheetname="Training_Data")
testing_data = pd.read_excel(file, sheetname="Test_Data")





num_layers = 3
num_in_layer = 8
numInputs = 5
numOutputs = 4
totalNodes = numInputs + numOutputs + num_layers * num_in_layer
outputNodes = list(range(totalNodes - numOutputs, totalNodes))
print outputNodes
idealOutput = {}
learning_rate = .1
numEpochs = 100
expected_cat = "very_low"
correctNode = totalNodes - 1

net = nx.Graph()
init_network(num_layers, num_in_layer, numInputs, numOutputs)

errors = []

print "pre-training results"
test_epoch_data(False, True)

print "training..."
#draw_network(net, num_layers, num_in_layer, numInputs, numOutputs)



for i in range(1, numEpochs + 1):

	errors.append(train_epoch_data())
	
	if i % (numEpochs / 20) == 0:

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
plt.plot(errors)
plt.ylabel('average Error per input')
plt.show()

print"100% complete"


print "Performance on Training Data"
test_epoch_data(False, True)

print "Performance on Testing Data"
test_epoch_data(True, False)

draw_network(net, num_layers, num_in_layer, numInputs, numOutputs)

print_results(True)

#for i in range(0, len(net.nodes())):
#	print net.node[i]


#draw_network(net, num_layers, num_in_layer, numInputs, numOutputs)
