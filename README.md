# C implementation of FRNN 

This document describes in detail the functionality of the **FRNN** model so it can be re-implemented in any programming language.

## Layers
The basic component of the model is a “layer”. Each layer receives an array of floating point values, performs a computation, and returns an output array of floating point values. The output dimensionality might be different from the input dimensionality and depends in a deterministic way on the input dimensionality.

The following figure gives a basic overview of all the layers of the model:


### Types of Layers

#### Input layer
Simply accepts an input vector.

#### Size X Convolutional layer
    
`Function`: Performs a size `X` convolution with `Y` filters. The input is a set of `Z` 1D arrays of length `L`. Each filter has dimensionality `X` by `Z`. I.e. it is a convolution over `X` elements of all `Z` input arrays simultaneously. The output of the i’th out of `Y` filters is a single number in the i’th output array.

##### Illustration of convolution operation.

 Each element is fed through an activation function which might be the identity, sigmoid, tanh or RELU functions. We use RELU:
<a href="https://www.codecogs.com/eqnedit.php?latex=$y&space;=&space;max(0,x)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$y&space;=&space;max(0,x)$" title="$y = max(0,x)$" /></a>. 
 
We consider only the ‘valid’ output elements, i.e. where the filter is fully within the input arrays. That means for a filter of size X = 2x + 1, the output array will have length (`L` input array) - 2x. 
`Parameters`: `Y` filters, each of dimensionality `X` by `Z`.

#### Max Pooling Layer
`Function`: Performs pooling over `X` elements at a time. The output array has size (L input array)//X. We consider `X` elements at a time. The output is the maximum over those `X` elements. Then we move our window by `X` and consider the next consecutive `X` elements, etc.


#### Concatenation Layer
`Function`: Concatenates the two input arrays.


#### LSTM layer
`Function`: Implements the functionality of an LSTM cell with `Y` neurons. See here for the equivalent python implementation. Output is an array of size `Y`. See equations here (“LSTM with forget gates”).
`Parameters`: For an input array of size `X`, and an LSTM cell with `Y` neurons, we have
Matrices `Wf`, `Wi`, `Wo`, `Wc` of size `Y` by `X`.
Matrices `Uf`, `Ui`, `Uo`, `Uc` of size `Y` by `Y`
Vectors `bf`, `bi`, `bo`, `bc` of size `Y`


#### Additional input:
The hidden state `h` (vector of size `X`) from the last timestep. All zeros if uninitialized (beginning of shot).

The state and output of the LSTM shall be updated at cycle t using the following equations:
<a href="https://www.codecogs.com/eqnedit.php?latex=f_t=g(W_f\cdot&space;x_t&plus;U_f\cdot&space;h_t-1&plus;b_{f}),&space;\\&space;i_t=g(W_i\cdot&space;x_t&plus;U_i\cdot&space;h_t-1&plus;b_{i}),&space;\\&space;o_t=g(W_o\cdot&space;x_t&plus;U_o\cdot&space;h_t-1&plus;b_{o}),&space;\\&space;c_t=f_t\cdot&space;c_t-1&plus;itc(Wcx_t&plus;Uch_t-1&plus;b_{c}),&space;\\&space;h_t=o_t\cdot&space;h(c_t)," target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_t=g(W_f\cdot&space;x_t&plus;U_f\cdot&space;h_t-1&plus;b_{f}),&space;\\&space;i_t=g(W_i\cdot&space;x_t&plus;U_i\cdot&space;h_t-1&plus;b_{i}),&space;\\&space;o_t=g(W_o\cdot&space;x_t&plus;U_o\cdot&space;h_t-1&plus;b_{o}),&space;\\&space;c_t=f_t\cdot&space;c_t-1&plus;itc(Wcx_t&plus;Uch_t-1&plus;b_{c}),&space;\\&space;h_t=o_t\cdot&space;h(c_t)," title="f_t=g(W_f\cdot x_t+U_f\cdot h_t-1+b_{f}), \\ i_t=g(W_i\cdot x_t+U_i\cdot h_t-1+b_{i}), \\ o_t=g(W_o\cdot x_t+U_o\cdot h_t-1+b_{o}), \\ c_t=f_t\cdot c_t-1+itc(Wcx_t+Uch_t-1+b_{c}), \\ h_t=o_t\cdot h(c_t)," /></a>

where xtRX is the input vector to the LSTM unit, ftRY is the ‘forget gate’ activation vector, itRY is the ‘input gate’ activation vector, otRY is the ‘output gate’ activation vector, htRY is the output vector of the LSTM unit (static), and ctRY is the cell state vector (static). The subscript t-1 refers to the value on the previous cycle.

The activation functions are defined as follows:
<a href="https://www.codecogs.com/eqnedit.php?latex=$$g(x)=\frac{1}{1-\exp(-x))}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$g(x)=\frac{1}{1-\exp(-x))}$$" title="$$g(x)=\frac{1}{1-\exp(-x))}$$" /></a>
(sigmoid function)
<a href="https://www.codecogs.com/eqnedit.php?latex=$$c(x)=tanh(x)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$c(x)=tanh(x)$$" title="$$c(x)=tanh(x)$$" /></a>
(hyperbolic tangent)

#### Fully Connected Layer
`Function`: Input of size `X` is matrix-multiplied by a `Y` by `X` matrix. The result is optionally fed through a nonlinear “activation” function such as sigmoid, RELU or tanh. Output has dimensionality `Y`.
`Parameters`: `Y` by `X` matrix of floats.
