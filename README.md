# Ratios model

The code is meant for inspiration and guidance - not as a tool - but dont hesitate to ask questions.
General infrastructure is needed for data loading, training, validation, testing, and interpreting. 

I think the model may be improved by:

(1) Adding ReLU activation after the input layer to ensure non-negative values in the ratios.

(2) Using log ratios instead of regular ratios.

(3) Using L1 regularization (lasso penalty instead of Ridge) in layer 1.

(4) Making the model more sparse - less parameters.

