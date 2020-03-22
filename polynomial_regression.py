import matplotlib.pyplot as plt
import numpy as np
import _pickle as pickle


class PolynomialRegression:

    def __init__(self, K):
        assert 1 <= K <= 10, f"K must be between 1 and 10. Got: {K}"
        self.K = K

        # Remember that we have K weights and 1 bias.
        self.parameters = np.ones((K + 1, 1), dtype=np.float)

    def predict(self, X):

        """ This method predicts the output of the given input data using the model parameters.
        Recall that a polynomial function is defined as:

        Given a single scalar input x,
        f(x) = w_0 + w_1 * x + w_2 * x^2 + ... + w_K * x^K

        Args:
        - X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar input data.

        Output:
        - ndarray (shape: (N, 1)): A N-column vector consisting N scalar output data.
        """
        assert X.shape == (X.shape[0], 1)

        # ===================================================

        # Obtain the K-th degree polynomial as a Matrix with the shape (len(train_X), K) with all ones
        poly_x = np.ones((len(X), self.K))
        # Raise col to obtain a polynomial basis function in the form as seen in class
        for i in range(0, self.K):
            poly_x[:, i] = np.power(np.transpose(X), self.K - i)

        # model.fit will get the parameters needed to predict Y based off of X using matrix multiplication
        parameters = self.parameters
        # Extract the parameters except for the 1st term
        weights = np.delete(parameters, 0, axis=0)
        # First term is the bias
        bias = parameters[0][0]
        # Use vector form of equation form class to predict a value of Y based off of bias and weights
        pred_Y = np.matmul(poly_x, weights) + bias

        return pred_Y

        # ====================================================

    def fit(self, train_X, train_Y):
        """ This method fits the model parameters, given the training inputs and outputs.

        Recall that the optimal parameters are:
        parameters = (X^{T}X)^{-1}X^{T}Y

        NOTE: Do not forget that we are using polynomial basis functions!

        Args:
        - train_X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training inputs.
        - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
        """
        assert train_X.shape == train_Y.shape and train_X.shape == (train_X.shape[0],
                                                                    1), f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_Y: {train_Y.shape})."
        assert train_X.shape[
                   0] >= self.K, f"require more data points to fit a polynomial (train_X: {train_X.shape}, K: {self.K}). Do you know why?"

        # ====================================================

        # Obtain the K-th degree polynomial as a Matrix with the shape (len(train_X), K) with all ones
        poly_x = np.ones((len(train_X), self.K + 1))
        # Raise col to obtain a polynomial basis function in the form as seen in class
        for i in range(0, self.K + 1):
            poly_x[:, -1 - i] = np.power(np.transpose(train_X), self.K - i)

        # Obtain the pinv. Using pinv because if det of poly_x^t * poly_x is 0 we will get a singular matrix
        pen_B = np.linalg.pinv(poly_x)
        # Compute optimal weights using formula from class
        w = np.matmul(pen_B, train_Y)

        self.parameters = w

        return self.parameters

        # ====================================================
        assert self.parameters.shape == (self.K + 1, 1)

    def fit_with_l2_regularization(self, train_X, train_Y, l2_coef):
        """ This method fits the model parameters with L2 regularization, given the training inputs and outputs.
        Recall that the optimal parameters are:
        parameters = (X^{T}X + lambda*I)^{-1}X^{T}Y2

        NOTE: Do not forget that we are using polynomial basis functions!

        Args:
        - train_X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training inputs.
        - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
        - l2_coef (float): The lambda term that decides how much regularization we want.
        """
        assert train_X.shape == train_Y.shape and train_X.shape == (train_X.shape[0],
        1), f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_Y: {train_Y.shape})."
        # ====================================================

        # Obtain the K-th degree polynomial as a Matrix with the shape (len(train_X), K) with all ones
        poly_x = np.ones((len(train_X), self.K + 1))
        # Raise col to obtain a polynomial basis function in the form as seen in class
        for i in range(0, self.K + 1):
            poly_x[:, -1 - i] = np.power(np.transpose(train_X), self.K - i)

        # Calculate w_bar by using formula (X^{T}X + lambda*I)^{-1}X^{T}Y2
        # Get size of identity matrix required then proceed with regularization
        B_2 = np.matmul(np.transpose(poly_x), poly_x)
        Id_size = np.ma.size(B_2, 0)
        Id = np.identity(Id_size)
        # Compute the m-p pinv with regularization
        reg_B = B_2 + l2_coef * Id

        # If coeff is 0, there will be no regularization of the optimal weights

        # Apply formula from class to obtain the weight
        inv_B = np.linalg.inv(reg_B)
        p_inv = np.matmul(inv_B, np.transpose(poly_x))
        w_bar = np.matmul(p_inv, train_Y)

        # Form a vector for the parameters
        self.parameters = w_bar

        return self.parameters

         # ====================================================

        assert self.parameters.shape == (self.K + 1, 1)

    def compute_mse(self, X, observed_Y):
        """ This method computes the mean squared error.

        Args:
        - X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar inputs.
        - observed_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar observed outputs.

        Output:
        - (float): The mean squared error between the predicted Y and the observed Y.
        """
        assert X.shape == observed_Y.shape and X.shape == (
            X.shape[0], 1), f"input and/or output has incorrect shape (X: {X.shape}, observed_Y: {observed_Y.shape})."
        pred_Y = self.predict(X)
        return ((pred_Y - observed_Y) ** 2).mean()

if __name__ == "__main__":
    # You can use linear regression to check whether your implementation is correct.
    # NOTE: This is just a quick check but does not cover all cases.
    model = PolynomialRegression(K=1)
    train_X = np.expand_dims(np.arange(10), axis=1)
    train_Y = np.expand_dims(np.arange(10), axis=1)

    model.fit(train_X, train_Y)
    optimal_parameters = np.array([[0.], [1.]])
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    pred_Y = model.predict(train_X)
    print("Correct predictions: {}".format(np.allclose(pred_Y, train_Y)))

    # Change parameters to suboptimal for next test
    model.parameters += 1
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    # Regularization pulls the weights closer to 0.
    optimal_parameters = np.array([[0.0231303], [0.99460293]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0.5)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))
