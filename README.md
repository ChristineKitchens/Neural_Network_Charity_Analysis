# Neural_Network_Charity_Analysis
A neural network model is implemented using the TensorFlow library to assess the impact of nonprofit donations.

## Overview
## Results
### Data PreProcessing
- Target Variable
- Model Features
- Removed Variables
### Compiling, Training, and Evaluating the Model
- How many neurons, layers, and activation functions did you select for your neural network model, and why?
  - Three layers total were selected for the NNM (Two hidden layers and one output layer). The two hidden layers had 120 and 80 nodes, respectively. 
  - The two hidden layers used a ReLU activation function and the output layer used a sigmoid activation function. ReLU functions were chosen for the first two layers since it's ideal for looking at positive nonlinear input data for classification or regression (as is the case with our model features). The sigmoid function was chosen for the output layer because it normalizes values to a probability between 0 and 1. This is ideal for binary classification, as the case is with our target variables.
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
- Model Target Achieved?
- Steps Taken to Attemp to Increase Model Performance:
## Summary
- Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.