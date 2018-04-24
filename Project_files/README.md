Running the Model:

The Python code for the model is in Neural.py. This program takes no command line arguments, all are defined in the code itself. There are 4 different ways to run the model, a single training run, a parameter sweep, a monte carlo simulation, and as a single training eopch. These can be chosen by uncommenting one of the 4 functions at the very bottom of the file. The parameters for the single run can be changed by changing the values initialized above it. You will need to download or create certain files for the model to run correctly

* _Required Files and Directories_
	* user_knowledge.xls - Theexcel File containing all the training and testing data
	* plots/ - A folder to store the plots from a parameter sweep. The program may not create the plots if this folder does not already exist
	* MCplots/ - A folder to store the plots from a Monte Carlo simulation. The program may not create the plots if this folder does not already exist

Organization of the Code:

The File is separated into 5 main sections

1. General Neural Network Functions - These are functions acting on the net as a whole and are usable for any Neural Network Configuration
2. Learning Base - These are functions that occur after the model has been initialized and are the base of the model itself. These will also perform on data set with no modification
3. Knowledge Level Implementation - Functions that are specific to this data set but vital to the model running. These would need to be adjusted for any change in data set or learning task
4. Analysis - These are high level functions used to test and analyze the model
5. Main Declarations and calls - This is where the global variables are declared and the anlysis function chosen.


* _Other Files_
	* binaryFnctns.py - not necessary for current iterations. Functions used to set and run the model to classify various binary operations (and, or, xor, nor)
	* file.gv - an example print of the draw_network() function
