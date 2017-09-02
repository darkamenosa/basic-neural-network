import numpy as np
import matplotlib.pyplot as plt
import h5py


########################################
# Activation functions
########################################
def relu(Z):
    cache = Z
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.multiply(dA, np.int64(Z > 0))
    assert(dZ.shape == Z.shape)
    return dZ


def sigmoid(Z):
    cache = Z

    A = 1/(1 + np.exp(-Z))

    assert(A.shape == Z.shape)

    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    dZ = dA * (sigmoid(Z) * (1 - sigmoid(Z)))

    assert(dZ.shape == Z.shape)
    return dZ


########################################
# Data processing functions
########################################

def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


########################################
# Neural network functions
########################################

def initialize_parameters(layer_dims):
    parameters = {}

    L = len(layer_dims)  # 4
    # X -> L1 -> L2 -> L3
    for l in range(1, L):
        # This just a sample data
        # l in [1, 2, 3]

        # Target:
        # W1(L1, X)
        # b1(L1, 1)
        # W2(L2, L1)
        # b2(L2, 1)
        # W3(L3, L2)
        # b3(L3, 1)

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(2/layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[1]
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_mini_batches = m // mini_batch_size

    for k in range(num_complete_mini_batches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k : mini_batch_size * (k + 1)]
        mini_batch_y = shuffled_Y[:, mini_batch_size * k : mini_batch_size * (k + 1)]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_mini_batches:]
        mini_batch_y = shuffled_Y[:, mini_batch_size * num_complete_mini_batches:]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def linear_forward(A_prev, W, b):
    cache = (A_prev, W, b)
    Z = W.dot(A_prev) + b

    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    else:
        raise ValueError('{0} activation is not supported'.format(activation))

    assert(A.shape == Z.shape)  # check shape

    cache = (activation_cache, linear_cache)
    return A, cache


def model_forward_propagation(X, parameters):
    L = len(parameters) // 2  # number of layers, parameters contain [W, b]
    caches = []
    A = X

    # Ex: L = 3
    # for this loop: l in [0, 1]
    for l in range(L - 1):
        W = parameters['W' + str(l + 1)]  # W1, W2
        b = parameters['b' + str(l + 1)]  # b1, b2
        A_prev = A # X, A1

        A, cache = linear_activation_forward(A_prev, W, b, activation='relu')
        caches.append(cache)


    WL = parameters['W' + str(L)]  # W3
    bL = parameters['b' + str(L)]  # W3
    AL_prev = A  # Value of A is now A2
    AL, cache = linear_activation_forward(AL_prev, WL, bL, activation='sigmoid')
    caches.append(cache)

    return AL, caches


def compute_cross_entropy_cost(AL, Y):
    assert(AL.shape == Y.shape) # check shape to compute cost
    m = Y.shape[1]
    print(np.log(1 - AL))
    logprobs = np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))
    cost = -(1./m) * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
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
        raise ValueError('{0} activation is not supported'.format(activation))

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def model_backward_propagation(AL, Y, caches):

    assert(AL.shape == Y.shape)

    grads = {}

    L = len(caches)  # number of layers
    m = Y.shape[1]   # number of examples

    dZL = AL - Y     # Sigmoid derivative Z
    cacheL = caches[L - 1]  # Get cache of the last layer
    dA_prev, dWL, dbL = linear_backward(dZL, cacheL[1])

    grads['dW' + str(L)] = dWL
    grads['db' + str(L)] = dbL

    # Ex: L = 3
    # l in [1, 0]
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(dA_prev, current_cache, activation='relu')
        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # parameters contain only [W, b]

    # Ex: L = 3
    # l will be [0, 1, 2]
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters


def model(X, Y, layer_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
        beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    parameters = initialize_parameters(layer_dims)
    costs = []
    for i in range(num_epochs):
        # Define mini_batches
        mini_batches = random_mini_batches(X, Y, mini_batch_size)

        for mini_batch in mini_batches:
            mini_batch_X, mini_batch_Y = mini_batch
            # forward propagation
            AL, caches = model_forward_propagation(mini_batch_X, parameters)

            # compute cost
            cost = compute_cross_entropy_cost(AL, mini_batch_Y)

            # backward propagation
            grads = model_backward_propagation(AL, mini_batch_Y, caches)

            # update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0 and print_cost:
            print('cost after {0} epochs is: {1}'.format(i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters


#################################
#### Test case
#################################
def test_initialize_parameters():
    # Test initialze_parameters
    # Shape:
    #   X        ->    L1     ->     L2     ->      L3
    #  (5, 1)    ->   (5, 5)  ->    (3, 5)  ->      (1, 3)

    print('test_initialize_parameters start:')

    parameters = initialize_parameters([5, 5, 3, 1])
    assert(parameters['W1'].shape == (5,5))
    assert(parameters['b1'].shape == (5,1))
    assert(parameters['W2'].shape == (3,5))
    assert(parameters['b2'].shape == (3,1))
    assert(parameters['W3'].shape == (1,3))
    assert(parameters['b3'].shape == (1,1))

    print('test_initialize_parameters passed!')


def test_random_mini_batches():
    print('test_random_mini_batches start:')

    print('case 1, data_size = 128, batch_size = 64')
    X = np.random.randn(1, 128)
    Y = np.random.randn(1, 128)
    size = 64
    mini_batches = random_mini_batches(X, Y, size)
    assert(len(mini_batches) == 2)


    print('case 2, data_size = 130, batch_size = 64')
    X = np.random.randn(1, 130)
    Y = np.random.randn(1, 130)
    size = 64
    mini_batches = random_mini_batches(X, Y, size)
    assert(len(mini_batches) == 3)

    # Check last mini_batch_X
    assert(len(mini_batches[2][0][0]) == 2)

    # Check last mini_batch_Y
    assert(len(mini_batches[2][1][0]) == 2)

    print('test_random_mini_batches passed!')


def test_relu():
    print('test_relu start:')

    Z = np.array([5, -1, 0, 1, 0])
    A, _ = relu(Z)
    expected = np.array([5, 0, 0, 1, 0])

    assert((A == expected).all())

    print('test_relu passed!')


def test_sigmoid():
    print('test_sigmoid start:')

    Z = np.array([5, -1, 0, 1, 0])
    A, _ = sigmoid(Z)
    expected = 1/(1 + np.exp(-Z))

    assert((A == expected).all())

    print('test_sigmoid passed!')


def test_model_forward_propagation():
    print('test_model_forward_propagation start:')

    np.random.seed(0)

    X = np.random.randn(5, 10)  # 5 features, 10 examples
    Y = np.random.randint(2, size=(1, 10))

    layer_dims = [X.shape[0], 5, 3, 1]
    parameters = initialize_parameters(layer_dims)

    AL, caches = model_forward_propagation(X, parameters)

    assert(AL.shape == (1, 10))
    assert(len(caches) == 3)

    ## Reimplement forward propagation to check result
    #
    # Network graph:
    #
    # X       ->    Z1  A1    ->    Z2  A2   ->   Z3  A3
    # (5, 10)      W1(5, 5)        W2(3, 5)      W3(1, 3)

    W1 = parameters['W1']
    b1 = parameters['b1']
    Z1 = np.dot(W1, X) + b1
    A1 = np.multiply(Z1, np.int64(Z1 > 0))

    W2 = parameters['W2']
    b2 = parameters['b2']
    Z2 = np.dot(W2, A1) + b2
    A2 = np.multiply(Z2, np.int64(Z2 > 0))

    W3 = parameters['W3']
    b3 = parameters['b3']
    Z3 = np.dot(W3, A2) + b3
    A3 = 1/(1 + np.exp(-Z3))

    assert((AL == A3).all())

    print('test_model_forward_propagation passed!')

def test_compute_cross_entropy_cost():
    print('test_compute_cross_entropy_cost start:')

    AL = np.array([[0.6, 0.3, 0.7, 0.1]])  # AL (1, 4)
    Y = np.array([[1, 0, 0, 1]])  # Y(1, 4)

    cost = compute_cross_entropy_cost(AL, Y)

    # (-1./4) * ((1 * np.log(0.6) + (1 - 1) * np.log(1 - 0.6)) + (0 * np.log(0.3) + (1 - 0) * np.log(1 - 0.3)) + (0 * np.log(0.7) + (1 - 0) * np.log(1 - 0.7)) + (1 * np.log(0.1) + (1 - 1) * np.log(1 - 0.1)))

    assert(cost == 1.0935146162561762)

    print('test_compute_cross_entropy_cost passed!')


######################################################
#### Support test_model_backward_propagation functions
######################################################

def vector_to_dict(theta):
    # Convert vector to parameter, for network graph:
    #
    # X       ->    Z1  A1    ->    Z2  A2   ->   Z3  A3
    # (5, 10)      W1(5, 5)        W2(3, 5)      W3(1, 3)
    #              b1(5, 1)        b2(3, 1)      b3(1, 1)

    # return theta
    parameters = {}

    parameters['W1'] = theta[:25, :].reshape(5, 5)
    parameters['b1'] = theta[25:30, :].reshape(5, 1)
    parameters['W2'] = theta[30:45, :].reshape(3, 5)
    parameters['b2'] = theta[45:48, :].reshape(3, 1)
    parameters['W3'] = theta[48:51, :].reshape(1, 3)
    parameters['b3'] = theta[51:, :].reshape(1, 1)

    assert(parameters['W1'].shape == (5, 5))
    assert(parameters['b1'].shape == (5, 1))
    assert(parameters['W2'].shape == (3, 5))
    assert(parameters['b2'].shape == (3, 1))
    assert(parameters['W3'].shape == (1, 3))
    assert(parameters['b3'].shape == (1, 1))

    return parameters

def dict_to_vector(dict):
    count = 0
    for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        if count == 0:
            theta = dict[key].reshape(-1, 1)
            count = count + 1
        else:
            theta = np.concatenate((theta, dict[key].reshape(-1, 1)))
    return theta


def gradient_to_vector(grads):
    count = 0
    for key in ['dW1', 'db1', 'dW2', 'db2', 'dW3', 'db3']:
        if count == 0:
            vec = grads[key].reshape(-1, 1)
            count = count + 1
        else:
            vec = np.concatenate((vec, grads[key].reshape(-1, 1)))
    return vec



def gradient_check(parameters, gradients, X, Y, epsilon=1e-7):
    # Gradient check for network graph:
    #
    # X       ->    Z1  A1    ->    Z2  A2   ->   Z3  A3
    # (5, 10)      W1(5, 5)        W2(3, 5)      W3(1, 3)

    grads = gradient_to_vector(gradients)  # (n, 1)
    theta = dict_to_vector(parameters)     # (n, 1)

    number_of_parameters = theta.shape[0]
    J_plus = np.zeros((number_of_parameters, 1))
    J_minus = np.zeros((number_of_parameters, 1))
    grads_approx = np.zeros((number_of_parameters, 1))

    for i in range(number_of_parameters):
        theta_plus = np.copy(theta)
        theta_plus[i][0] = theta_plus[i][0] + epsilon
        AL_plus, _ = model_forward_propagation(X, vector_to_dict(theta_plus))
        J_plus[i][0] = compute_cross_entropy_cost(AL_plus, Y)

        theta_minus = np.copy(theta)
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        AL_minus, _ = model_forward_propagation(X, vector_to_dict(theta_minus))
        J_minus[i][0] = compute_cross_entropy_cost(AL_minus, Y)

        grads_approx[i][0] = (J_plus[i][0] - J_minus[i][0]) / (2 * epsilon)

    numerator = np.linalg.norm(grads_approx) - np.linalg.norm(grads)
    denominator = np.linalg.norm(grads_approx + grads)

    difference = numerator / denominator

    print('difference: ', difference)
    if difference <= 1e-7:
        print('Good backward propagation!')
    else:
        print('There is a problem with your backward propagation')

    return difference


def test_model_backward_propagation():
    print('test_model_backward_propagation start:')

    np.random.seed(0)

    X = np.random.randn(5, 10) / 255.
    Y = np.random.randint(2, size=(1, 10))

    layer_dims = [X.shape[0], 5, 3, 1]
    parameters = initialize_parameters(layer_dims)

    AL, caches = model_forward_propagation(X, parameters)
    grads = model_backward_propagation(AL, Y, caches)

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    dW3 = grads['dW3']
    db3 = grads['db3']

    assert(dW1.shape == (5, 5))
    assert(db1.shape == (5, 1))
    assert(dW2.shape == (3, 5))
    assert(db2.shape == (3, 1))
    assert(dW3.shape == (1, 3))
    assert(db3.shape == (1, 1))

    # Re-implement forward and backward propagation

    ## Reimplement forward propagation to check result
    #
    # Network graph:
    #
    # X       ->    Z1  A1    ->    Z2  A2   ->   Z3  A3
    # (5, 10)      W1(5, 5)        W2(3, 5)      W3(1, 3)

    # forward propagation
    W1 = parameters['W1']
    b1 = parameters['b1']
    Z1 = np.dot(W1, X) + b1
    A1 = np.multiply(Z1, np.int64(Z1 > 0))

    W2 = parameters['W2']
    b2 = parameters['b2']
    Z2 = np.dot(W2, A1) + b2
    A2 = np.multiply(Z2, np.int64(Z2 > 0))

    W3 = parameters['W3']
    b3 = parameters['b3']
    Z3 = np.dot(W3, A2) + b3
    A3 = 1/(1 + np.exp(-Z3))


    # backward propagation
    m = Y.shape[1]

    dZ3 = (A3 - Y)
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dZ2 = np.multiply(dA2, np.int64(Z2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dZ1 = np.multiply(dA1, np.int64(Z1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    assert(grads['dW3'].shape == dW3.shape)
    assert(grads['db3'].shape == db3.shape)
    assert(grads['dW2'].shape == dW2.shape)
    assert(grads['db2'].shape == db2.shape)
    assert(grads['dW1'].shape == dW1.shape)
    assert(grads['db1'].shape == db1.shape)

    assert((grads['dW3'] == dW3).all())
    assert((grads['db3'] == db3).all())
    assert((grads['dW2'] == dW2).all())
    assert((grads['db2'] == db2).all())
    assert((grads['dW1'] == dW1).all())
    assert((grads['db1'] == db1).all())


    # vec = gradient_to_vector(grads)
    theta = dict_to_vector(parameters)
    n_parameters = vector_to_dict(theta)

    assert((parameters['W1'] == n_parameters['W1']).all())
    assert((parameters['b1'] == n_parameters['b1']).all())
    assert((parameters['W2'] == n_parameters['W2']).all())
    assert((parameters['b2'] == n_parameters['b2']).all())
    assert((parameters['W3'] == n_parameters['W3']).all())
    assert((parameters['b3'] == n_parameters['b3']).all())
    gradient_check(parameters, grads, X, Y)

    print('test_model_backward_propagation passed!')


def test_update_parameters():
    print('test_update_parameters start:')

    parameters = {
        'W1': np.array([1, 2, 3, 4, 5]),
        'b1': np.array([0, 0, 0, 0, 0]),
        'W2': np.array([7, 8, 9]),
        'b2': np.array([1, 1, 1])
    }

    grads = {
        'dW1': np.array([1, 2, 3, 4, 5]),
        'db1': np.array([1, 1, 0, 1, 1]),
        'dW2': np.array([0.5, 0.6, 0.7]),
        'db2': np.array([0, 0, 0]),
    }

    learning_rate = 0.1
    new_parameters = update_parameters(parameters, grads, learning_rate)

    expected = {
        'W1': np.array([0.9, 1.8, 2.7, 3.6, 4.5]),
        'b1': np.array([-0.1, -0.1, 0, -0.1, -0.1]),
        'W2': np.array([6.95, 7.94, 8.93]),
        'b2': np.array([1, 1, 1])
    }

    assert((new_parameters['W1'] == expected['W1']).all())
    assert((new_parameters['b1'] == expected['b1']).all())
    assert((new_parameters['W2'] == expected['W2']).all())
    assert((new_parameters['b2'] == expected['b2']).all())

    print('test_update_parameters passed!')



def main():
    # test_initialize_parameters()
    # test_random_mini_batches()
    # test_relu()
    # test_sigmoid()
    # test_model_forward_propagation()
    # test_compute_cross_entropy_cost()
    test_model_backward_propagation()
    # test_update_parameters()


if __name__ == '__main__':
    main()
