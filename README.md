# C implementation of FRNN 

This document describes in detail the functionality of the FRNN model so it can be re-implemented in any programming language.

## Layers
The basic component of the model is a “layer”. Each layer receives an array of floating point values, performs a computation, and returns an output array of floating point values. The output dimensionality might be different from the input dimensionality and depends in a deterministic way on the input dimensionality.

The following figure gives a basic overview of all the layers of the model:


### Types of Layers

#### Input layer
Simply accepts an input vector.

#### Size X Convolutional layer
    
`Function`: Performs a size X convolution with Y filters. The input is a set of Z 1D arrays of length L. Each filter has dimensionality X by Z. I.e. it is a convolution over X elements of all Z input arrays simultaneously. The output of the i’th out of Y filters is a single number in the i’th output array.

##### Illustration of convolution operation.

 Each element is fed through an activation function which might be the identity, sigmoid, tanh or RELU functions. We use RELU (y = max(0,x)). We consider only the ‘valid’ output elements, i.e. where the filter is fully within the input arrays. That means for a filter of size X = 2x + 1, the output array will have length (L input array) - 2x. 
`Parameters`: Y filters, each of dimensionality X by Z.


#### Max Pooling Layer
`Function`: Performs pooling over X elements at a time. The output array has size (L input array)//X. We consider X elements at a time. The output is the maximum over those X elements. Then we move our window by X and consider the next consecutive X elements, etc.


#### Concatenation Layer
`Function`: Concatenates the two input arrays.


#### LSTM layer
`Function`: Implements the functionality of an LSTM cell with Y neurons. See here for the equivalent python implementation. Output is an array of size Y. See equations here (“LSTM with forget gates”).
`Parameters`: For an input array of size X, and an LSTM cell with Y neurons, we have
Matrices Wf, Wi, Wo, Wc of size Y by X.
Matrices Uf, Ui, Uo, Uc of size Y by Y
Vectors bf, bi, bo, bc of size Y


#### Additional input:
The hidden state h (vector of size X) from the last timestep. All zeros if uninitialized (beginning of shot).

The state and output of the LSTM shall be updated at cycle t using the following equations:
ft=g(Wfxt+Ufht-1+bf),
it=g(Wixt+Uiht-1+bi),
ot=g(Woxt+Uoht-1+bo),
ct=ftct-1+itc(Wcxt+Ucht-1+bc),
ht=oth(ct),

where xtRX is the input vector to the LSTM unit, ftRY is the ‘forget gate’ activation vector, itRY is the ‘input gate’ activation vector, otRY is the ‘output gate’ activation vector, htRY is the output vector of the LSTM unit (static), and ctRY is the cell state vector (static). The subscript t-1 refers to the value on the previous cycle.

The activation functions are defined as
g(x)=11+e-x=exex+1 (sigmoid function)
c(x)=tanh(x) (hyberbolic tangent)

#### Fully Connected Layer
`Function`: Input of size X is matrix-multiplied by a Y by X matrix. The result is optionally fed through a nonlinear “activation” function such as sigmoid, RELU or tanh. Output has dimensionality Y.
`Parameters`: Y by X matrix of floats.
