#class: training
#methods: cost function, error function, gradient descent

#start out for univariate regression model
#generalize to multivariate model

import numpy as np
import matplotlib as matp
import housing.data

class LinearRegression(object):
    '''A linear regression model that can be trained using multivariate linear regression'''

    def __init__(self, num_regressors = 1):
        '''Constructor
        :param num_regressors
        '''

        # total number of regressors to predict target variable
        self.num_regressors = num_regressors


    def initialise(self, initial_regressor coefficients):
        '''Initialise the coefficients of this model

        :param initial_regressor_coefficients: List of initial coefficients  for each regressor.)
        '''


    def train(self, data_path):
        '''Train the model on data.

        :param data_path: The path to the data file
        '''

    def prepare_data(data_path):
        '''Read and prepares raw data for use in other methods.

        :param data_path: Path to the raw data.
        :returns: observation: Vector containing information in data.'''
        #open file containing data in reading mode
        open(data_path,"r")
        
        for observation in data_path:
            read

    def cost_function(input,intersect, slope):
        '''Computes the sum of error squares for a hypothesis and a set of observations.

        :param input: Observation pairs from data.
        :param intersect: The intersection of the regression line with the y-axis.
        :param slope: The slope of the regression line.
        '''

        for each



