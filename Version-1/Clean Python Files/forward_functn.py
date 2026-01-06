#forward pass function

def forward_pass(X, params):
    cache = {}

    #layer 1:
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = relu(Z1)
    cache['Z1'] = Z1
    cache['A1'] = A1

    #layer 2:
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = relu(Z2)
    cache['Z2'] = Z2
    cache['A2'] = A2

    #layer 3:
    Z3 = np.dot(A2, params['W3']) + params['b3']
    A3 = relu(Z3)
    cache['Z3'] = Z3
    cache['A3'] = A3

    #OUTPUT LAYER:
    Z4 = np.dot(A3, params['W4']) + params['b4']
    A4 = softmax(Z4)
    cache['Z4'] = Z4
    cache['A4'] = A4

    return A4, cache
