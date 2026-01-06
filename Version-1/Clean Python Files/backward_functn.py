#backward pass function:


def backward_function(X, y_true, params, cache):


    grads = {}


    m = X.shape[0]


    dz4 = cache['A4'] - y_true
    grads['dw4'] = np.dot(cache['A3'].T, dz4) / m
    grads['db4'] = np.sum(dz4, axis=0, keepdims=True) / m   


    #layer 3
    dA3 = np.dot(dz4, params['W4'].T)
    dZ3 = dA3 * relu_derivative(cache['Z3'])
    grads['dw3'] = np.dot(cache['A2'].T, dZ3) / m
    grads['db3'] = np.sum(dZ3, axis=0, keepdims=True) / m


    #layer 2
    dA2 = np.dot(dZ3, params['W3'].T)
    dZ2 = dA2 * relu_derivative(cache['Z2'])
    grads['dw2'] = np.dot(cache['A1'].T, dZ2) / m
    grads['db2'] = np.sum(dZ2, axis=0, keepdims=True) / m


    #layer 1
    dA1 = np.dot(dZ2, params['W2'].T)
    dZ1 = dA1 * relu_derivative(cache['Z1'])
    grads['dw1'] = np.dot(X.T, dZ1) / m
    grads['db1'] = np.sum(dZ1, axis=0, keepdims=True) / m


    return grads

