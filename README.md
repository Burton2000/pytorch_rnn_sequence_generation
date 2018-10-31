# RNN time series generation experiments
This repo provides an example of training an RNN/GRU model to overfit and generate a sin wave that it's trained on. Once trained we supply an initial sequence of points and the model will then continue to generate the sin wave its learnt. 

Obviously this isn't a particularly difficult task but is meant as just a way to demonstrate simple use of RNNs in PyTorch.

## How to use
* Run train.py file, your model will be trained and then will automatically generate points after training.  

## This is the result after training.
![Alt text](/generated_wave.png?raw=true "Sin wave generation results")

# Updated for PyTorch 0.4.1
