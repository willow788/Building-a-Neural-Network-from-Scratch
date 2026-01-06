#testing the model on test data
X_test_reshaped = X_test_norm.reshape(X_test_norm.shape[0], 28*28)
y_test_reshaped = y_test_encoded

test_predictions = predict(X_test_reshaped, params)
test_accuracy = np.sum(test_predictions == np.argmax(y_test_reshaped, axis=1)) / X_test_reshaped.shape[0]
print(f"Test Accuracy: {test_accuracy:.4f}")
