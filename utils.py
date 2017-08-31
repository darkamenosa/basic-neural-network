import numpy as np
import h5py
import matplotlib.pyplot as plt


def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu(Z):
    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def load_data():

    train_dataset = h5py.File('train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def initialize_parameters_deep(layer_dims):
    parameters = {}

    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2/layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A_prev, W, b):
    # Z = W.A_prev + b
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    else:
        raise ValueError('Activation {0} is not supported'.format(activation))

    cache = (activation_cache, linear_cache)

    assert(A.shape == Z.shape)

    return A, cache


def L_model_forward(X, parameters, debug=False):

    caches = []
    L = len(parameters) // 2  # parameters only contain W, b
    if debug:
        print('number of forward layers: ', L)
    A = X
    for l in range(L - 1):
        if debug:
            print('Compute forward propagation for layer: ', l + 1)
            print('Compute forward propagation for W{0}, b{1} '.format(l + 1, l + 1))
        W = parameters['W' + str(l + 1)]
        b = parameters['b' + str(l + 1)]
        A_prev = A
        A, cache = linear_activation_forward(A_prev, W, b, 'relu')
        caches.append(cache)
    if debug:
        print('Compute forward propagation for layer: ', L)
        print('Compute forward propagation for W{0}, b{1} '.format(L, L))

    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, WL, bL, 'sigmoid')
    caches.append(cache)

    return AL, caches


def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    # dZ.shape == (L, m)
    # W.shape  == (L, L - 1)
    # b.shape  == (L, 1)
    # A_prev.shape == (L - 1, m)


    # dW = dZ x A_prev.T (L, L - 1)
    # dA_prev = W.T x dZ (L - 1, m)

    dW = (1./m) * np.dot(dZ, A_prev.T)
    db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    activation_cache, linear_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError('Activation {0} is not supported'.format(activation))

    dA_prev, dW, db  = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, debug=False):
    grads = {}
    L = len(caches)

    if debug:
        print('numers of backward layers:', L)
        print('compute backward propagation for layer: ', L)

    # dAL = Y/AL - (1 - y)/(1 - AL)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    cacheL = caches[L - 1]
    dA_prev, dWL, dbL = linear_activation_backward(dAL, cacheL, 'sigmoid')

    grads['dW' + str(L)] = dWL
    grads['db' + str(L)] = dbL

    for l in reversed(range(L - 1)):
        # l = [1, 0]
        if debug:
            print('compute backward propagation for layer: ', l + 1)
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(dA_prev, current_cache, 'relu')

        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db

    return grads


def linear_backward_with_regularization(dZ, cache, lamb):
    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = (1./m) * np.dot(dZ, A_prev.T) + (lamb/m) * W
    db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db


def linear_activation_backward_with_regularization(dA, cache, lamb, activation):
    activation_cache, linear_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError('Activation {0} is not supported'.format(activation))

    dA_prev, dW, db = linear_backward_with_regularization(dZ, linear_cache, lamb)

    return dA_prev, dW, db


def L_model_backward_with_regularization(AL, Y, caches, lamb, debug=False):
    grads = {}
    L = len(caches)

    if debug:
        print('numers of backward layers:', L)
        print('compute backward propagation for layer: ', L)

    # dAL = Y/AL - (1 - y)/(1 - AL)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    cacheL = caches[L - 1]
    dA_prev, dWL, dbL = linear_activation_backward_with_regularization(dAL, cacheL, lamb, 'sigmoid')

    grads['dW' + str(L)] = dWL
    grads['db' + str(L)] = dbL

    for l in reversed(range(L - 1)):
        # l = [1, 0]
        if debug:
            print('compute backward propagation for layer: ', l + 1)
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward_with_regularization(dA_prev, current_cache, lamb, 'relu')

        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db

    return grads


def compute_cost(AL, Y):
    # Logistic regression cost
    # cost = -1./m * sum(Y*log(AL) + (1 - Y)*log(1 - AL))
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
    return cost


def compute_cost_with_regularization(AL, Y, parameters, lamb):
    m = Y.shape[1]
    # print('compute cost with: ', m, 'examples')
    cross_entropy_cost = compute_cost(AL, Y)
    # print('cross_entropy_cost: ', cross_entropy_cost)

    L2_regularization_cost = lamb/(2.*m)

    L = len(parameters) // 2

    Wsum = 0
    for l in range(L):
        W = parameters['W' + str(l + 1)]
        Wsum = Wsum + np.sum(np.square(W))

    L2_regularization_cost = L2_regularization_cost * Wsum

    # print("L2_regularization_cost: ", L2_regularization_cost)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # parameters only store W, b

    for l in range(L):
        # print('dW' + str(l + 1), grads['dW' + str(l + 1)])
        # print('db' + str(l + 1), grads['db' + str(l + 1)])
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters

def L_model_with_regularization(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, lamb=0.1, print_cost=False):
    parameters = initialize_parameters_deep(layer_dims)

    costs = []
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost_with_regularization(AL, Y, parameters, lamb)
        # print('cost', cost)
        grads = L_model_backward_with_regularization(AL, Y, caches, lamb)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0 and print_cost:
            print('Cost after {0} iterations is: {1}'.format(i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()
    return parameters


def L_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    parameters = initialize_parameters_deep(layer_dims)

    costs = []
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0 and print_cost:
            print('Cost after {0} iterations is: {1}'.format(i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()
    return parameters


#############################################
#### Dropout implementation
#############################################


def linear_activation_forward_with_dropout(A_prev, W, b, keep_prob, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    else:
        raise ValueError('Activation {0} is not supported'.format(activation))

    D = np.random.rand(A.shape[0], A.shape[1])
    D = D < keep_prob
    A = np.multiply(A, D)
    A = A/keep_prob

    cache = (activation_cache, linear_cache, D)

    print('activation: ', activation)


    # implement dropout here

    assert(A.shape == Z.shape)

    return A, cache


def L_model_forward_with_dropout(X, parameters, keep_prob):
    caches = []
    L = len(parameters) // 2  # parameters contain only W, b

    A = X
    for l in range(L - 1):
        # l = [1, 0]
        W = parameters['W' + str(l + 1)]
        b = parameters['b' + str(l + 1)]
        A_prev = A
        A, cache = linear_activation_forward_with_dropout(A_prev, W, b, keep_prob, 'relu')
        caches.append(cache)

    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    AL, cache = linear_activation_forward_with_dropout(A, WL, bL, 1, 'sigmoid')
    caches.append(cache)

    return AL, caches


def linear_activation_backward_with_dropout(dA, cache, keep_prob, activation):
    activation_cache, linear_cache, D = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError('Activation {0} is not supported'.format(activation))

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    dA_prev = np.multiply(dA_prev, D)
    dA_prev = dA_prev/keep_prob
    return dA_prev, dW, db

def L_model_backward_with_dropout(AL, Y, caches, keep_prob):
    grads = {}
    L = len(caches)  # number of layers

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    cacheL = caches[L - 1]

    dA_prev, dWL, dbL = linear_activation_backward_with_dropout(dAL, cacheL, keep_prob, 'sigmoid')
    grads['dW' + str(L)] = dWL
    grads['db' + str(L)] = dbL

    for l in reversed(range(L - 1)):
        # l [1, 0]
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward_with_dropout(dA_prev, current_cache, keep_prob, 'relu')

        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db

    return grads


def L_model_with_dropout(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, keep_prob=0.86, print_cost=False):
    parameters = initialize_parameters_deep(layer_dims)

    costs = []

    for i in range(num_iterations):
        AL, caches = L_model_forward_with_dropout(X, parameters, keep_prob)
        cost = compute_cost(AL, Y)
        grads = L_model_backward_with_dropout(AL, Y, caches, keep_prob)
        if i % 100 == 0 and print_cost:
            print('cost after {0} iterations is: {1}', i, cost)
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters



#############################################
#### Gradient check
#############################################


def gradient_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def dict_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys


def vector_to_dict(theta):

    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters



def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Build gradient check for layer_dims [5, 5, 3, 1]

    Returns:
        [float] -- return the difference of grad and grad_approx
    """
    # Network shape:
    #
    # X        a1         a2         a3
    # 5   ->   5    ->    3    ->    1
    #
    # W1 (5, 5), b1 (5, 1) # W2 (3, 5), b2 (3, 1)
    # W3 (1, 3), b3 (1, 1)

    grads = gradient_to_vector(gradients)
    theta, _ = dict_to_vector(parameters)

    assert(grads.shape == theta.shape)

    J_plus = np.zeros(theta.shape)
    J_minus = np.zeros(theta.shape)
    grads_approx = np.zeros(theta.shape)

    number_of_parameters = theta.shape[0]

    for i in range(number_of_parameters):
        theta_plus = np.copy(theta)  # shape (n, 1)
        theta_plus[i][0] = theta_plus[i][0] + epsilon

        AL_plus, _ = L_model_forward(X, vector_to_dict(theta_plus))
        J_plus[i][0] = compute_cost(AL_plus, Y)

        theta_minus = np.copy(theta)
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        AL_minus, _ = L_model_forward(X, vector_to_dict(theta_minus))
        J_minus[i][0] = compute_cost(AL_minus, Y)

        grads_approx[i][0] = (J_plus[i][0] - J_minus[i][0])/(2 * epsilon)
        print(i, grads[i][0], grads_approx[i][0], "diff: ", grads_approx[i][0] - grads[i][0])


    numerator = np.linalg.norm(grads_approx - grads)
    denominator = np.linalg.norm(grads_approx) + np.linalg.norm(grads)

    difference = np.divide(numerator, denominator)
    if difference > 1e-7:
        print('There is a mistake in your backward propagation! difference = ' + str(difference))
    else:
        print('Your backward propagation work perfectly fine! difference = ' + str(difference))
    return difference



def predict(X, y, parameters):
    probs, _ = L_model_forward(X, parameters)
    correct = (probs > 0.5) == y
    return correct.sum() / y.shape[1]


def main():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = load_data()
    # X = (train_set_x_orig.reshape(train_set_x_orig.shape[0], -1) / 255.).T
    # Y = (train_set_y_orig.reshape(train_set_y_orig.shape[0], 1)).T
    # X_test = (test_set_x_orig.reshape(test_set_x_orig.shape[0], -1) / 255.).T
    # Y_test = (test_set_y_orig.reshape(test_set_y_orig.shape[0], 1)).T

    # parameters = L_model(X, Y, [X.shape[0], 4, 3, 1], learning_rate=0.0075, num_iterations=3000, print_cost=True);

    # acc_train = predict(X, Y, parameters)
    # acc_test = predict(X_test, Y_test, parameters)

    # print('Accuracy on train set', acc_train)
    # print('Accuracy on test set', acc_test)


    np.random.seed(0)

    # X = np.random.randn(4, 10)
    # Y = np.array([[1, 1, 0, 0, 1, 1, 0, 0, 1, 0]])

    # parameters = initialize_parameters_deep([X.shape[0], 5, 3, 1])
    # AL, caches = L_model_forward(X, parameters, True)

    # cost = compute_cost(AL, Y)
    # print('\ncost: ', cost, '\n')

    # grads = L_model_backward(AL, Y, caches, True)

    # gradient_check_n(parameters, grads, X, Y)



if __name__ == '__main__':
    main()
