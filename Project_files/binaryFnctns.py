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

