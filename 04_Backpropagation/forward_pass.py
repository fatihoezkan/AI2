import numpy as np

# Xaiver init.

def xavier_init(shape = tuple):
    if len(shape) == 1:
        number_of_in, number_of_out = shape[0], shape[0]
    elif len(shape) == 2:
        number_of_in, number_of_out = shape[0], shape[1]
    else:
        raise ValueError("Xavier initialization supports only 1D and 2D shapes.")

    scaler= np.sqrt(2.0 / (number_of_in + number_of_out))

    return np.random.normal(loc=0.0, scale=scaler, size=shape)

def bias(shape = tuple):
    bias = np.random.normal(loc=0.0, size = shape[1] )
    return bias


#activation function
def sigmoid(input): 
     return (1 / (1+np.exp(-1 * input)))

#activision funct derivative
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))


#LOSS FUNC
def Loss_derivative(target_output , output):
    return (output - target_output) / (output * (1-output))

def Loss(target_output , output):
    return ((-target_output * np.log(output)) - ((1-target_output) * np.log(1-output)))


#gradient - backpropagation
def bias_grad(x):
    return 1.

def weight_deriv(x):
    pass




