# Model Proposal for _[Project Name]_

_Your Name_

* Course ID: CMPLXSYS 530,
* Course Title: Computer Modeling of Complex Systems
* Term: Winter, 2018



&nbsp; 

### Goal 
*****
 
For this project I will be constructing a neural network using agent based modeling. I am doing this to become more familiar with the underlying mechanisms of neural networks and to examine how different stuctural make-ups and design choices affect the effectiveness of the learning process. This will all be done by training and testing the neural network on classifying sentences into different categories.

&nbsp;  
### Justification
****
I am using agent based modeling for this project because each I believe it is a very intuitive way to create the network in which each interacting neuron has very simple behaviors. These neurons will be simply deciding whether how strong its output should be based on the input it recieves from the neurons around it, and each neuron does this using the same criteria. This interaction between the neurons and simple decisions of the neurons make this ripe for Agent Based Modeling.

&nbsp; 
### Main Micro-level Processes and Macro-level Dynamics of Interest
****

A properly implemented neural network will "learn" how to recognize patterns and classify data. This is done by training the network with specific training data using alternating cycles of forward propagation, in which the network is given data and attempts to classify it, and back propagation, in which the weights of neuron connections are updated based on the error between the expected value and the actual output of the network from the forward propagation. After the training is done the network should be able to classify similar data, even if it has not seen that exact before. The accuracy and efficiency of this learning is the process that I am interested in studying.

Specifically, I am looking to examine how the structure of the network and certain parameters such as learning rate or the number of training passes will affect this learning. Example of these effects could be undertraining, in which the network does not learn how to classify the training data let alone the testing data, or overtraining, which prevents the network from classifying any data that deviates at all from the training data.

&nbsp; 


## Model Outline
****
&nbsp; 
### 1) Environment
The environment in this model is fairly limited. It consists of the groupings of agents which separates them into layers, and whether the network is being trained or tested. It will also contain the rate at which the connection between neurons are allowed to change.

* _List of environment-owned variables_
	* state - whether the network is training or being tested
	* layers - number of hidden layers
* _List of environment-owned methods/procedures_
	* set_training - set the environment to a training state, where the network performs both forward and back propgataion
	* set_testing - set the environment to a testing state, where the network performs only forward propagation


```python
# Include first pass of the code you are thinking of using to construct your environment
# This may be a set of "patches-own" variables and a command in the "setup" procedure, a list, an array, or Class constructor
# Feel free to include any patch methods/procedures you have. Filling in with pseudocode is ok! 
# NOTE: If using Netlogo, remove "python" from the markdown at the top of this section to get a generic code block
```

&nbsp; 

### 2) Agents
 
 The agents in this model are the individual neurons. There will be 3 different types of neurons considered- input, hidden, and output neurons. Hidden and output neurons will contain a list of neurons that they receive input from and and the weight of these connections. They will be arranged in groups called layers and each neuron in a given layer will receive input from every neuron in the previous layer. The output neurons are the final layer and there output will be considered the output of the whole system. The input neurons are the most uniwue. Their output will be determined by the input data, so they will not rely on any previous activation values
 
 
* _List of agent-owned variables_
	* input_neurons  - a list of neurons that give input to this neuron
	* input_weights - a list of the connection weights between the neurons listed in input_neurons and this neuron
	* summed_input - the total sum of all the weighted inputs to this neuron
	* output - the activation value of this neuron
	* error - The measure of error that this neuron contributed to an incorrect answer, used for back-propagation
* _List of agent-owned methods/procedures_
	* activate - take summed_input and use it to find the activation value of this neuron
	* sum_inputs - calculate summed_input
	* get_output - report the output of this neuron
	* calc_error - calculates the error of this neuron
	* update_weight - updaates the weight of this neuron's connections



```python
# Include first pass of the code you are thinking of using to construct your agents
# This may be a set of "turtle-own" variables and a command in the "setup" procedure, a list, an array, or Class constructor
# Feel free to include any agent methods/procedures you have so far. Filling in with pseudocode is ok! 
# NOTE: If using Netlogo, remove "python" from the markdown at the top of this section to get a generic code block
```

&nbsp; 

### 3) Action and Interaction 
 
**_Interaction Topology_**

Each agent in the input layer will be set by the data and will interact with the agents in the first hidden layer. All agents in the hidden layers will recieve input from every neuron in the previous layer and will send their output to neurons in the following layer. The final layer of neurons will report their output, and this will be compared to the expected output. This process is called forward propagation. While the model is in the training state, this error will then be used to adjust the weight the connections between neurons. The error of each neuron is dependent on the neurons in the layers succeeding them, so this information is sent in the opposite directon as it was in forward propagation and is called back propagation. 
 
**_Action Sequence_**

_What does an agent, cell, etc. do on a given turn? Provide a step-by-step description of what happens on a given turn for each part of your model_

Forward Propagation

1. Agents find the weighted activation value of each neuron in the previous layer by multiplying the neuron's activation value by the weight of the connection between the two.
2. These weighted activation values are summed together
3. This sum is then put through a sigmoid function to determine the activation value of the neuron, which is then set and ready to be accessed by neurons in the following layer

Backward Propagation for output neurons

1. Calculate the squared difference of the expected value to the neuron's output
2. Calculate the derivative of the neuron's output
3. Multiply this difference and the derivative find the error introduced by this neuron
4. update the weight of the neuron's input connections using the formula weight = weight + learning_rate * error * input

Backward Propagation for hidden neurons

1. Calculate the weighted error of each neuron connected to this one in the succeeding row by multiplying its error by the weight of the connection between the neurons
2. Sum these errors together
2. Calculate the derivative of the neuron's output
3. Multiply this sum and the derivative to find the error introduced by this neuron
4. update the weight of the neuron's input connections using the formula weight = weight + learning_rate * error * input

&nbsp; 
### 4) Model Parameters and Initialization

_Describe and list any global parameters you will be applying in your model._
	* learning_rate - the rate at which the connections are allowed to change. Higher rates mean that the connections will make more drastic adjustments in each cycle while lower rates mean they will make smaller.
	* num_epochs - An epoch refers to when the network is exposed to each item in the training set exactly once. This is the number of times that the network will be exposed to each training data point during the training phase
	* training_data - This will contain the data that the network is trained on.
	* testing_data - this will contain the data that the network is tested on

_Describe how your model will be initialized_
	*


_Provide a high level, step-by-step description of your schedule during each "tick" of the model_
	* Training
		1. Set the input neurons based on the given data for this turn
		2. Forward propagate to get the network's output
		3. Compare the network's output to the expected output
		4. If they are different, back propagate the error and adjust the weights
	* Testing
		1. Set the input neurons based on the given data for this turn
		2. Forward propagate to get the network's output
		3. Compare the network's output to the expected output and record whether they match
&nbsp; 

### 5) Assessment and Outcome Measures

_What quantitative metrics and/or qualitative features will you use to assess your model outcomes?_
* percentage of correctly classified training data points
* percentage of correctly classified testing data points

By examining these two outcomes, I will be able to determine how well the network has been trained in its current configuration. By testing it separately on both testing and training data I can determine if the network has been overtrained or undertrained. A perfectly trained network would get a very high percentage correct in both tests, an undertrained network will have a much lower percentage correct for both, an overtrained network would be have a very high percentage on the training data and a very low percentage on the testing data.

&nbsp; 

### 6) Parameter Sweep

_What parameters are you most interested in sweeping through? What value ranges do you expect to look at for your analysis?_

* number of hidden layers - I will sweep between having 0 layers and 10 layers (a network with 0 hidden layers would simply have the input neurons connected directly to the output neurons)
* number of neurons per layer - I will sweep between having 1 neuron and 10 neurons per layer
* learning rate for the connections - This value can range from 0 to 1, I will sweep through this in increments of .01
* number of epochs - This value will range from 1 to 100 either in intervals of 1 or 2
