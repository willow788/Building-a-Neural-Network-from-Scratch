#weight update function
def update_weights(params, grads, learn_rate):
    params['W1'] -= learn_rate * grads['dw1']
    params['b1'] -= learn_rate * grads['db1']

    params['W2'] -= learn_rate * grads['dw2']
    params['b2'] -= learn_rate * grads['db2']

    params['W3'] -= learn_rate * grads['dw3']
    params['b3'] -= learn_rate * grads['db3']

    params['W4'] -= learn_rate * grads['dw4']
    params['b4'] -= learn_rate * grads['db4']

    return params
