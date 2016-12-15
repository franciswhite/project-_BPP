#class: training
#methods: cost function, error function, gradient descent

#start out for univariate regression model
#generalize to multivariate model

import numpy as np
import matplotlib as matp

class LinearRegression(object):
    '''A linear regression model that can be trained using multivariate linear regression'''

    def __init__(self, num_regressors = 1):
        '''Constructor
        :param num_regressors
        '''

        # Number of regressors to predict target variable
        self.num_regressors = num_regressors
        pass

    def initialise(self, initial_regressor_coefficients = None):
        '''Initialise the coefficients of this model

        :param initial_regressor_coefficients: List of initial coefficients  for each regressor.)
        '''
        pass

    def train(self, data_path):
        '''Train the model on data.

        :param data_path: The path to the data file
        '''
        pass

    def prepare_data(data_path, model):
        '''Read and prepares raw data for use in other methods.

        :param data_path: Path to the raw data.
        :returns: datapoints: Array containing information in textfile.'''

        #Determine number of observations
        global number_observations
        number_observations = 0
        with open(data_path) as data:
            for line in data:
                number_observations += 1
        number_observations
        print(number_observations) #DEBUGGING


        with open(data_path) as data:
            temp_list= []
            for line in data:
                line = line.split() # to deal with blank
                if line:            # lines (ie skip them)
                    line = [float(i) for i in line]
                    temp_list.append(line)

        datapoints = np.array(temp_list)
        return datapoints

        # print(datapoints)

    def cost_function(array,intersect, slope):
        '''Computes the average sum of error square for a hypothesis and a set of observations.

        :param array: Observation pairs (arrays) from data.
        :param intersect: The intersection of the regression line with the y-axis.
        :param slope: The slope of the regression line.

        :return cost: Sum of error squares divided by 2*#observations
        '''

        #Sum
        total_error = 0
        for row in range(0,number_observations-1):
            single_observation_error = ((intersect + slope * array[row][0]) - array[row][1])**2
            total_error += single_observation_error

        #Cost function as suggested by Andrew Ng
        cost = 1/(2*number_observations)*total_error
        return cost

    def minimize(array):
        '''Minimizes cost function using gradient descent.

        :param: array: data with respect to which cost function should be minimized
        :returns: Array containing local minima for arguments.
        '''
        #Initialize total error
        total_error = 0

        #Initialize arguments as 0
        coefficient0 = 0
        coefficient1 = 0

        #Initialize learning rate: YET TO TEST WHAT'S BEST
        alpha = 0.001

        #Initialize update variables
        temp0 = 10000000000000
        temp1 = 10000000000000

        #Initialize output variable

        #Convergence margin
        epsilon = 0.5
        x=0
        while True:
            for row in range(0, number_observations-1): #This loops calculates sum of squared errors for parameters in iteration
                single_observation_error = ((coefficient0 + coefficient1 * array[row][0]) - array[row][1])**2
                total_error += single_observation_error

                for row in range(0, number_observations-1): #This loops calculates the updates and performs them
                    temp0 = coefficient0 - alpha * (1/number_observations) * total_error
                    temp1 = coefficient1 - alpha * (1/number_observations) * total_error * array[row][0]

                    if abs(coefficient0 - temp0) > epsilon and abs(coefficient1 - temp1) > epsilon: #Convergence condition
                        coefficients = np.array([coefficient0, coefficient1]) #Assemble array
                        break #Stop the while loop
                    else:
                        coefficient0 = temp0 #Update coefficients
                        coefficient1 = temp1
                        total_error = 0 #Resets total_error before next iteration
            print(x)
            print(coefficient0)
            print(coefficient1)
            x+=1
        return coefficients



scaffold = LinearRegression() #Initialize LinearRegression object with desired number of regressors

a = LinearRegression.prepare_data("housing.data", scaffold)

b = LinearRegression.cost_function(a, 0, 1.5)

okay = LinearRegression.minimize(a)

print(okay)

