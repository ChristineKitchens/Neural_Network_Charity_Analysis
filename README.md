# Neural Network Charity Analysis
## Overview
A neural network model (NNM) is implemented using the TensorFlow library to assess the impact of nonprofit donations. Each donation and associated project is coded 0 or 1 depending on whether the funded project was deemed "successful" or not. A script for the [initial model](Notebooks/AlphabetSoupCharity.h5) is documented in the [AlphabetSoupCharity.ipynb](Notebooks/AlphabetSoupCharity.ipynb) file. The goal was to create an optimized version of the model where the accuracy met or exceeded 75%. The [AlphabetSoupCharity_Optimization.ipynb](Notebooks/AlphabetSoupCharity_Optimization.ipynb) contains the modified, optimized script of the original model.
## Results
### Data PreProcessing
- Target Variable(s)
  - IS_SUCCESSFUL - Was the money used effectively
- Model Features
  - APPLICATION_TYPE - Alphabet Soup application type
  - AFFILIATION - Affiliated sector of industry
  - CLASSIFICATION - Government organization classification
  - USE_CASE - Use case for funding
  - ORGANIZATION - Organization type
  - INCOME_AMT - Income classification
  - ASK_AMT - Funding amount requested
- Removed Variable(s)
  - EIN - Identification column
  - NAME -  Identification column
  - STATUS - Active status
  - SPECIAL_CONSIDERATIONS - Special consideration for application
### Compiling, Training, and Evaluating the Model
- Three layers total were selected for the NNM (Two hidden layers and one output layer). Three layers were chosen because, after experimenting with various amounts of layers, two hidden layers and one output layer yielded the maximum amount of accuracy increase.
- The two hidden layers had 120 and 80 nodes, respectively. A general rule of thumb is that the number of nodes in the input layer should be equal to 2-3 times the number of input features. Given that the encoded data frame being input to the model contained 41 columns/features, that amount was multiplied by 3 to identify an approximate number of nodes.
- The two hidden layers used a ReLU activation function and the output layer used a sigmoid activation function. ReLU functions were chosen for the first two layers since it's ideal for looking at positive nonlinear input data for classification or regression (as is the case with our model features). The sigmoid function was chosen for the output layer because it normalizes values to a probability between 0 and 1. This is ideal for binary classification, as the case is with our target variables.
- Exact script containing model definitions is located below.
```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
hidden_nodes_layer1 = 120
hidden_nodes_layer2 = 80

nn_opt1 = tf.keras.models.Sequential()

# First hidden layer
nn_opt1.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn_opt1.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn_opt1.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn_opt1.summary()
```
#### Model Evaluation
- Unfortunately, the [optmized model](Notebooks/AlphabetSoupCharity_Optimization.h5) did not meet the 75% accuracy criteria.
- The following steps were taken to attempt to increase the model performance:
  - The STATUS and SPECIAL_CONSIDERATIONS variables were dropped (in addition to the initial EIN and NAME variables) to remove noisy variables
  - The number of nodes in each hidden layer was increased
  - A third hidden layer was added
  - Three hidden layers were used, node amounts were increased, and the activation function in the third hidden layer was changed from ReLU to tanh.
  - While not detailed in the [AlphabetSoupCharity_Optimization.ipynb](Notebooks/AlphabetSoupCharity_Optimization.ipynb) file, other modifications were attempted, including dropping the CLASSIFICATION variable, modifying bin sizes for the CLASSIFICATION and APPLICATION_TYPE variables, various combinations of node numbers, etc. Despite various attempts at optmization, the accuracy never exceeded %72.56.
## Summary
- Despite numerous attempts at optimization, the model did not meet the required 75% accuracy threshold. In fact, despite various changes, the accuracy never deviated far from the %72.40 of the original, unoptimized model.
- Given the numerous failed attempts to improve the efficiency of the charity NNM, a different model entirely may be needed. A random forest (RF) classifier is a good alternative. Given that the charity data is in a tabular form, an RF model would be compatible with the data as is. The RF model also tends to provide comparable accuracy at greater speed and with less code. The decreased processing time would also better facilitate ongoing attempts at optimization since each iteration would not need the lengthy reprocessing time required of the NNM.

## Resources
- Software
  - Jupyter Notebook
- Notebooks
  - [AlphabetSoupCharity.ipynb](Notebooks/AlphabetSoupCharity.ipynb)
  - [AlphabetSoupCharity_Optimization.ipynb](Notebooks/AlphabetSoupCharity_Optimization.ipynb)
- Data
  - [charity_data.csv](Resources/charity_data.csv)
- Models
  - [AlphabetSoupCharity.h5](Notebooks/AlphabetSoupCharity.h5)
  - [AlphabetSoupCharity_Optimization](Notebooks/AlphabetSoupCharity_Optimization.h5)