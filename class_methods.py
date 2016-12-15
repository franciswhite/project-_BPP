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
        #Initialize errors sums
        error_sum_0 = 0
        error_sum_1 = 0
        #Initialize arguments as 0
        coefficient0 = 0
        coefficient1 = 0

        #Initialize learning rate: YET TO TEST WHAT'S BEST
        alpha = 0.00001

        #Initialize update variables
        temp0 = 10000
        temp1 = 10000

        #Convergence margin
        epsilon = 0.00000001
        stop = "run" #
        x = 1 #DEBUGGING
        while stop != "Break the while loop.":
            for row in range(0, number_observations): #Loop computes possible update of coefficient0
                single_observation_error = ((coefficient0 + coefficient1 * array[row][0]) - array[row][1])
                error_sum_0 += single_observation_error
            for row in range(0, number_observations): #This loop computes possible update coefficient1
                single_observation_error_times_observation = ((coefficient0 + coefficient1 * array[row][0]) - array[row][1])*array[row][0]
                error_sum_1 += single_observation_error_times_observation
            temp0 = coefficient0 - alpha * (1/number_observations) * error_sum_0
            temp1 = coefficient1 - alpha * (1/number_observations) * error_sum_1
            if abs(coefficient0 - temp0) < epsilon and abs(coefficient1 - temp1) < epsilon: #Convergence condition
                coefficient0 = temp0
                coefficient1 = temp1
                coefficients = np.array([coefficient0, coefficient1]) #Assemble array
                stop = "Break the while loop." #Stop the while loop
            else: #If convergence not yet reached
                coefficient0 = temp0 #Update coefficients
                coefficient1 = temp1
                error_sum_0 = 0 #Resets total_errors before next iteration
                error_sum_1 = 0
        return coefficients



scaffold = LinearRegression() #Initialize LinearRegression object with desired number of regressors

a = LinearRegression.prepare_data("toy_data.py", scaffold)

b = LinearRegression.cost_function(a, 0, 1)

okay = LinearRegression.minimize(a)
print(okay)

