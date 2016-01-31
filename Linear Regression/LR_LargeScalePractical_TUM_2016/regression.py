#!/usr/bin/python
# Name: Chua Ruiwen Raymond
# Date: 31/1/2016
# Matriculation No. : 03669295
# Task: Implement a least squares regression using gradient descent
# Implementation Algorithm based on Andrew Ng's Linear Regression Notes in Coursera
# Source: https://d396qusza40orc.cloudfront.net/ml/docs/slides/Lecture4.pdf


filename = "airfoil_self_noise.dat"     #train data


def main():

    M = sum(1 for line in open(filename))   #num rows in data

    feature1 = []       #array for features
    feature2 = []
    feature3 = []
    feature4 = []
    feature5 = []

    features = []

    Y = []              #array for train target values.

    with open(filename) as f:
        for line in f:

            # split the text
            values = line.split()

            feature1.append(float(values[0]))
            feature2.append(float(values[1]))
            feature3.append(float(values[2]))
            feature4.append(float(values[3]))
            feature5.append(float(values[4]))

            Y.append(float(values[5]))

    #Apply feature scaling to speed up gradient descent. feature0 is just an array of 1.
    feature0_norm = [1] * len(feature1)
    feature1_norm = normalize(feature1)
    feature2_norm = normalize(feature2)
    feature3_norm = normalize(feature3)
    feature4_norm = normalize(feature4)
    feature5_norm = normalize(feature5)

    features = [feature0_norm, feature1_norm, feature2_norm, feature3_norm, feature4_norm, feature5_norm]

    # theta(weight) vector, using values from the slides as a starting point
    theta = [130, 0, 0, -35, 0, -150]

    # convergence threshold
    threshold = 0.001

    # learning rate
    alpha = 0.003

    #Perform linear regression to get the final theta values
    weights = linearRegression(theta, features, Y, threshold, alpha)

    newWeights = unnormalize(weights, [feature0_norm,feature1, feature2, feature3, feature4, feature5])
    print(newWeights)


    #perform testing by comparing with the training data
    results1 = 0
    results2 = 0

    for m in range(0, M):


        #My Model
        temp1 = newWeights[0] + (newWeights[1] * feature1[m]) + (newWeights[2] * feature2[m]) + (newWeights[3] * feature3[m]) \
               + (newWeights[4] * feature4[m]) + (newWeights[5] * feature5[m])


        #Model from the slide
        temp2 = 132.83 + (-0.00013 * feature1[m]) + (-0.4221 * feature2[m]) + (-35.6471 * feature3[m]) \
               + (0.099 * feature4[m]) + (-147.29 * feature5[m])

        diff1 = abs(Y[m] - temp1)

        diff2 = abs(Y[m] - temp2)

        if diff1 <= 0.1:
            results1 += 1

        if diff2 <= 0.1:
            results2 += 1

    print("b:", newWeights[0])
    print("w:", newWeights[1:6])
    print("My Model's Accuracy: ", str((results1/(M*1.0))*100))
    print("Model in slides's Accuracy: ", str((results2/(M*1.0))*100))


'''
Method to un-normalise the theta(weights) based on the features.
For every feature, get the max value.
New weight value = max value * current weight value.

Input:
weights- > An array of weight values
feature -> An array of features. Each feature is an array itself.

Output: An array of weight values.
'''


def unnormalize(weights, features):
    numFeature = len(features)
    newWeight = []

    for i in  range(0, numFeature):
        currentFeature = features[i]
        maxVal = max(currentFeature)

        newWeight.append(weights[i] / maxVal)

    return newWeight


'''
Method to normalise the features to speed up the gradient descent.
For every feature array, get the max value and the length of the array.
New feature value = current feature value / max value

Input:
feature -> An array of features. Each feature is an array itself.

Output: An array of normalized values between(0.0 to 1.0)
'''


def normalize(feature):
    size = len(feature)
    maxVal = max(feature)


    normF = []

    for i in range(0, size):
        normF.append(feature[i] / maxVal)

    return normF


'''
Method to calculate the cost function for every row in the data.
Cost Function = 1/2*M*(H_theta(x_i)-y(i))^2, where M = no. rows in data, i = current row in data, H_theta = theta * x

Input:
theta -> An array of theta values
features-> An array of features
Y -> An array of target values

Output: An array of normalized values between(0.0 to 1.0)
'''


def costFunction(theta, features, Y):
    costVal = 0
    N = len(features)  # num of features
    M = len(features[0])  # num of training rows

    for row in range(0, M):

        H = 0

        for featureIndex in range(0, N):

            currentFeature = features[featureIndex]

            H += (currentFeature[row] * theta[featureIndex])

        costVal += pow((H - Y[row]), 2)

    return costVal / (2 * M)


'''
Method to update the theta values.
New theta values(j) = old theta values(j) - (alpha * (1/M * (H_theta(x(i)-y(i)) * x(j)))), where alpha = the learning rate,
i = current row in data, M = num of rows in data, j current theta index.

Input:
theta -> An array of theta values
features-> An array of features
Y -> An array of target values
alpha -> learning rate

Output: An array of new theta values
'''


def updateTheta(theta, features, Y, alpha):
    N = len(features)  # num of features
    M = len(features[0])  # num of training rows

    newTheta = []

    for i in range(0, N):

        costVal = 0

        for row in range(0, M):

            H = 0

            x_j = features[i][row]

            for featureIndex in range(0, N):
                currentFeature = features[featureIndex]
                H += (currentFeature[row] * theta[featureIndex])

            costVal += ((H - Y[row]) * x_j)

        total = alpha * (costVal / M)

        currentTheta = theta[i] - total

        newTheta.append(currentTheta)

    return newTheta

'''
Method to perform linear regression and obtain the optimal theta values.
Stop when new cost function - old cost function < threshold.
New cost function is computed with updated theta values.


Input:
theta -> An array of theta values
features-> An array of features
Y -> An array of target values
threshold -> value to determine convergence
learningRate -> learning rate

Output: An array of new theta values
'''


def linearRegression(theta, features, Y, threshold, learningRate):

    bestTheta = []

    while True:

        J_old = costFunction(theta, features, Y)
        theta = updateTheta(theta, features, Y, learningRate)
        J_new = costFunction(theta, features, Y)

        diff = abs(J_new - J_old)

        print "diff: ", diff

        if diff <= threshold:
            bestTheta = theta
            break

    return bestTheta


if __name__ == "__main__":
    main()
