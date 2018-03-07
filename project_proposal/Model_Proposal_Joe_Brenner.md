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

A properly implemented neural network will "learn" how to recognize patterns and classify data. This is done by training the network with specific training data, and after the training is done it should be able to classify similar data, even if it has not seen that exact before. The accuracy and efficiency of this learning is the process that I am interested in studying. I am looking to examine how the structure of the network and the nature of the training data will affect this learning process. 

&nbsp; 


## Model Outline
****
&nbsp; 
### 1) Environment
The environment in this model is fairly limited. It consists of the groupings of agents which separates them into layers, and whether the network is being trained or tested. It will also contain the rate at which the connection between neurons are allowed to change.

* _List of environment-owned variables
state
layer_size
num_layers
learning_rate
* _List of environment-owned methods/procedures
set_training
set_testing


```python
# Include first pass of the code you are thinking of using to construct your environment
# This may be a set of "patches-own" variables and a command in the "setup" procedure, a list, an array, or Class constructor
# Feel free to include any patch methods/procedures you have. Filling in with pseudocode is ok! 
# NOTE: If using Netlogo, remove "python" from the markdown at the top of this section to get a generic code block
```

&nbsp; 

### 2) Agents
 
 The agents in this model are the individual neurons. There will be 3 different types of neurons considered- input, hidden, and output neurons. Hidden and output neurons will contain a list of neurons that they receive input from and and the weight of these connections. They will be arranged in groups called layers and each neuron in a given layer will receive input from every neuron in the previous layer. The output neurons are the final layer and there output will be considered the output of the whole system. The input neurons are the most unique. Their output will be determined by the input data, so they will not rely on any previous activation values.
 
 
* _List of agent-owned variables
* input_neurons (list)
* input_weights (list)
* summed_input (double)
* output (double)
* error (double)

* _List of agent-owned methods/procedures
activate
sum_inputs
get_output
update_weight
calc_error



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

_Describe how your model will be initialized_

_Provide a high level, step-by-step description of your schedule during each "tick" of the model_

&nbsp; 

### 5) Assessment and Outcome Measures

_What quantitative metrics and/or qualitative features will you use to assess your model outcomes?_

&nbsp; 

### 6) Parameter Sweep

_What parameters are you most interested in sweeping through? What value ranges do you expect to look at for your analysis?_
