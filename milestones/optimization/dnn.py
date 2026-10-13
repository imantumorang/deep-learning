import numpy as np
from typing import NamedTuple

def relu(Z):
    a = np.maximum(0,Z)
    return a

def sigmoid(Z):
    a = 1 / (1 + np.exp(-Z))
    return a


def initialize_parameters (layer_dims):
    L = len(layer_dims) 
    # print(f"Len Param {len(layer_dims)}")
    np.random.seed(7)
    parameters = {}
    for l in range (1, L): # start from layer 1, we don't have W0 or b0, we have weights and bias from layer 1
        # print (f"l={l}")
        parameters["W"+ str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters



class ActivationCache (NamedTuple):
    Z: np.ndarray

def activate(Z, activation_function):
    cache = ActivationCache(Z = Z)
    if (activation_function == "relu"):
        A = relu(Z)
    elif (activation_function == "sigmoid"):
        A = sigmoid(Z)
    
    return A, cache

class PropagateCache (NamedTuple):
    W: np.ndarray
    Aprev: np.ndarray
    b: np.ndarray

def propagate(W, Aprev, b):
  cache = PropagateCache(W=W, b=b, Aprev=Aprev)
  Z = np.dot(W,Aprev) + b
  return Z, cache

class ForwardCache (NamedTuple):
    PropagateCache: PropagateCache
    ActivationCache: ActivationCache


def AL_derivative(AL,Y):
    # since our cost function uses binary cross-entropy, then:
    # -(Y/A) + ((1-Y)/ (1-A))
    dAL = -(np.divide(Y,AL)) + np.divide((1-Y), (1-AL))
    return dAL


# In backward, we will pass the forward_pass cache, from which we can extract
# the activation cache.
# ReLU backward function logic:
# if Z <= 0 return 0; else return dA
# Note: We must use an element-wise condition because Z and dA are arrays.
# Since this is used in hidden layers, we accept dA from the backward sequence
# and apply the chain rule.
def relu_backward(dA, activation_cache):
    Z = activation_cache.Z
    dZ = np.array(dA, copy=True) # Create a copy to safely modify the array
    # When Z <= 0, the local derivative is 0, so set dZ to 0 (element-wise)
    dZ[Z <= 0] = 0
    return dZ

# Sigmoid backward function logic:
# First calculate the local derivative of the activation function with respect to Z:
# dLocal = A * (1 - A)
# then apply the chain rule with the incoming gradient (dA) to compute dZ
def sigmoid_backward(dA, activation_cache):
    Z = activation_cache.Z

    # Get the original activation output matrix 'A'
    A = sigmoid(Z)
    
    # Calculate the local derivative and apply the chain rule
    dZLocal = A * (1 - A)
    dZ = dA * dZLocal

    return dZ
    
# Propagate backward: calculate the derivative of  Z = W . Aprev + b
def propagate_backward(dZ, Aprev, W):
    m = Aprev.shape[1]
    dW = 1/m * np.dot(dZ, Aprev.T)
    # sum horizontally across the batch dimension, keeping it as a column vector
    # because we only have 1 bias per neurons
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dAprev = np.dot(W.T, dZ)
    return dW, db, dAprev
