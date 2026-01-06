#defining the predict function

def predict(X, params):
    A4, _ = forward_pass(X, params)
    predictions = np.argmax(A4, axis=1)
    return predictions
