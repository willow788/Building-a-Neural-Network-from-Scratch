#relu activation function
def relu(x):
    return np.maximum(0, x)

#derivative of relu
def relu_derivative(x):
    return (x>0).astype(float)

#softwax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#cross entropy loss function
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]

    log_likelihood = -np.sum(y_true * np.log(y_pred + 1e-9), axis=1)
    loss = np.sum(log_likelihood) / m
    return loss

