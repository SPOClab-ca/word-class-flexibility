import sklearn.linear_model
from scipy import stats, linalg
import numpy as np
import pandas as pd

# https://gist.github.com/fabianp/9396204419c7b638d38f
def calculate_partial_correlation(input_df):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables,
    controlling for all other remaining variables

    Parameters
    ----------
    input_df : array-like, shape (n, p)
        Array with the different variables. Each column is taken as a variable.

    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of input_df[:, i] and input_df[:, j]
        controlling for all other remaining variables.
    """
    partial_corr_matrix = np.zeros((input_df.shape[1], input_df.shape[1]));
    for i, column1 in enumerate(input_df):
        for j, column2 in enumerate(input_df):
            control_variables = np.delete(np.arange(input_df.shape[1]), [i, j]);
            if i==j:
                partial_corr_matrix[i, j] = 1;
                continue
            data_control_variable = input_df.iloc[:, control_variables]
            data_column1 = input_df[column1].values
            data_column2 = input_df[column2].values
            fit1 = sklearn.linear_model.LinearRegression(fit_intercept=True)
            fit2 = sklearn.linear_model.LinearRegression(fit_intercept=True)
            fit1.fit(data_control_variable, data_column1)
            fit2.fit(data_control_variable, data_column2)
            residual1 = data_column1 - (np.dot(data_control_variable, fit1.coef_) + fit1.intercept_)
            residual2 = data_column2 - (np.dot(data_control_variable, fit2.coef_) + fit2.intercept_)
            partial_corr_matrix[i,j] = stats.pearsonr(residual1, residual2)[0]
    return pd.DataFrame(partial_corr_matrix, columns = input_df.columns, index = input_df.columns)
