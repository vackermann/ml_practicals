import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
import scipy.stats
import collections
from operator import add
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class NBC():
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
        self.distributions = {}
    
    def fit(self, X_train, y_train):
        y_class_xvals_dict = {}
        y_class_distribution_dict = {}
        
        # Put all y classes present in y_train into y_class dicts
        for yval in y_train:
            if yval not in y_class_xvals_dict.keys():
                y_class_xvals_dict[yval]=[]
                y_class_distribution_dict[yval] =[]
    
        # Add all training instances to their respective y val in class dict
        for i in range(len(X_train)):
            xval = X_train[i]
            y_class_xvals_dict[y_train[i]].append(xval)

        # Calculate the distribution of each feature...
        for f_index, feature_type in enumerate(self.feature_types):
            
            # ... for each class depending on feature type
            for yval in y_class_xvals_dict.keys():

                # Get list of all x_feature values for respective y class
                feature_xvals_for_y = []
                full_xvals_for_y = y_class_xvals_dict[yval]
                for xval in full_xvals_for_y:
                    feature_xvals_for_y.append(xval[f_index])

                distribution_for_yval = []

                # Distribution computation depends on feature type:
                # -> for real-valued features: Gaussian mean and standard deviation
                if feature_type == 'r':
                    mean = np.mean(feature_xvals_for_y)
                    distribution_for_yval.append(mean)
                    std = np.std(feature_xvals_for_y)
                    # Standard deviation must not be zero (otherwise divion by zero error during prediction)
                    if std == 0:
                        std = 1e-6
                    distribution_for_yval.append(std)

                # -> for binary features: Bernoulli random variables
                if feature_type == 'b':
                    counter = collections.Counter(feature_xvals_for_y)
                    
                    # Bernoulli parameter must not be 0 or 1 -> Laplace smoothing
                    p = (counter[1.0]+1 )/ (len(feature_xvals_for_y)+2)
                    distribution_for_yval.append(p)
                    
                y_class_distribution_dict[yval].append(distribution_for_yval)
        self.distributions = y_class_distribution_dict
        return y_class_distribution_dict
    
    def predict(self, X):
        y_predictions = []

        # Iterate over all values in the test set and find the y for which the respective xval has the highest probability
        for x_index in range(X.shape[0]):
            # lower limit for y class
            y_with_max_prob = -1
            max_prob = -1

            # Get probability for every y value
            for y_val in self.distributions.keys():
                # Initial probablity of y value 
                y_prob = 1

                # Factor in indiviudual probability of each feature for this y val
                for f_index, feature_type in enumerate(self.feature_types):

                    # Computation of feature probability depends on feature type
                    # -> real-valued feature
                    if feature_type == 'r':
                        mean = self.distributions[y_val][f_index][0]
                        std = self.distributions[y_val][f_index][1]
                        feature_prob = scipy.stats.norm(mean, std).pdf(X[x_index,f_index])

                    # -> binary feature
                    if feature_type == 'b':
                        p = self.distributions[y_val][f_index][0]
                        if X[x_index,f_index] == 0.0:
                            feature_prob = 1-p
                        elif X[x_index,f_index] == 1.0:
                            feature_prob = p

                    # Factor feature prob into overall prop of this y value
                    y_prob *= feature_prob

                # Override current prediction favorite if prop of this y value is higher
                if (y_prob>max_prob):
                    max_prob = y_prob
                    y_with_max_prob = y_val

            # Add prediction for this x val to prediction lists
            y_predictions.append(y_with_max_prob)
        return y_predictions

def trainNBCOnIncreasingSetAndReturnAccuracy(num_steps, feature_types, num_classes, X_train, y_train, X_test, y_test):
    accuracys = []
    N = X_train.shape[0]
    for k in range(1, num_steps+1, 1):
        set_size = int(N * (k/num_steps))
        X_k_train = X_train[:set_size]
        y_k_train = y_train[:set_size]
        nbc = NBC(feature_types, num_classes)
        nbc.fit(X_k_train, y_k_train)
        yhat = nbc.predict(X_test)
        accuracy = np.mean(yhat == y_test)
        accuracys.append(accuracy)
    return accuracys

def trainLogRegOnIncreasingSetAndReturnAccuracy(num_steps, X_train, y_train, X_test, y_test):
    accuracys = []
    N = X_train.shape[0]
    for k in range(1, num_steps+1, 1):
        set_size = int(N * (k/num_steps))
        X_k_train = X_train[:set_size]
        y_k_train = y_train[:set_size]
        lor = LogisticRegression(solver='lbfgs')
        lor.fit(X_k_train, y_k_train)
        accuracy = lor.score(X_test, y_test)
        accuracys.append(accuracy)
    return accuracys

def testModelsOnMultiplePermutationsAndSetSizes(dataset_name, num_permuations, feature_types, num_classes, X, y):
    N, D = X.shape
    Ntrain = int(0.8 * N)
    shuffler = np.random.permutation(N)
    num_steps = 10

    sum_accuracy_nbc = [0] * num_steps
    sum_accuracy_lor = [0] * num_steps

    for i in range(0, num_permuations, 1):
        print("%i out of %i done." %(i, num_permuations))
        # Shuffle data
        shuffler = np.random.permutation(N)
        Xtrain = X[shuffler[:Ntrain]]
        ytrain = y[shuffler[:Ntrain]]
        Xtest = X[shuffler[Ntrain:]]
        ytest = y[shuffler[Ntrain:]]

        # get accuracy scores
        accuracy_nbc = trainNBCOnIncreasingSetAndReturnAccuracy(num_steps, feature_types, num_classes, Xtrain, ytrain, Xtest, ytest)
        accuracy_lor = trainLogRegOnIncreasingSetAndReturnAccuracy(num_steps, Xtrain, ytrain, Xtest, ytest)

        sum_accuracy_nbc = list( map(add, sum_accuracy_nbc, accuracy_nbc) )
        sum_accuracy_lor = list( map(add, sum_accuracy_lor, accuracy_lor) )

    average_accuracy_nbc = list([(x/num_permuations) for x in sum_accuracy_nbc]);
    average_accuracy_lor = list([(x/num_permuations) for x in sum_accuracy_lor]);

    plotLearningCurvesTwoModels(dataset_name, num_permuations, average_accuracy_nbc, average_accuracy_lor, num_steps)
    return

def plotLearningCurvesTwoModels(dataset_name, num_permuations, accuracy_nbc, accuracy_lor, num_steps):
    fig, ax = plt.subplots()

    size_percent_list = []
    for k in range(1, num_steps+1, 1):
        size_percent = 100 * (k / num_steps)
        size_percent_list.append(size_percent)

    #training_sizes = range(9,109,10)
    ax.plot(size_percent_list, accuracy_nbc)
    ax.plot(size_percent_list, accuracy_lor)
    ax.set(xlabel='Training dataset size in %', ylabel='Average accuracy over '+str(num_permuations)+" permutations" ,
           title=dataset_name +':\n Classification accuracy against percentage of data used for training.')
    ax.grid()
    ax.set_ylim([0.55,1.05])
    ax.set_xlim([0,100])
    ax.legend(["Naive Bayes Classifier", "Logistic Regression"])

    fig.savefig(dataset_name+".png")

    #plt.show()
    return

if __name__ == '__main__':

    # Iris dataset

    iris = load_iris()
    X, y = iris['data'], iris['target']
    feature_types_iris = ['r', 'r', 'r', 'r']
    num_classes_iris = 3
    dataset_name_iris = "Iris"

    testModelsOnMultiplePermutationsAndSetSizes(dataset_name_iris,250, feature_types_iris, num_classes_iris, X, y)

    # Voting dataset 

    X, y = cp.load(open('voting.pickle', 'rb'))
    feature_types_voting = ['b'] * 16
    num_classes_voting = 2
    dataset_name_voting = "US Congressional Voting"
    
    testModelsOnMultiplePermutationsAndSetSizes(dataset_name_voting,250, feature_types_voting, num_classes_voting, X, y)

