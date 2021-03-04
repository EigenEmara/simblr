# simblr
A Simple Bayesian Linear Regression Library.

simblr is a toy library to implement probabilistic linear regression methods in "Mathematics for Machine Learning" book by Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong.

The code implements the following linear regression algorithms:
1. Maximum likelihood estimation.
2. Maximum a posteriori
3. Bayesian linear regression.

## Example Output
Produced by `test.py`
1. MLE fits a 5th order polynomial (highly overfitting).
2. MAP applies a Gaussian prior to the basis function parameters (similar to L2 regularization).
3. Bayes linear regression provides a distribution over functions plotted by dark and light gray confidence intervals.

![simblr_output](https://user-images.githubusercontent.com/70483607/109917013-49938580-7cbd-11eb-9f80-c4fc3760946d.png)

