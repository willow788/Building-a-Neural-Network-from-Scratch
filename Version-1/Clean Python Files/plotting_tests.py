#plotting 
import matplotlib.pyplot as plt

idx = np.random.randint(0, len(X_test_norm))
sample_img = X_test_norm[idx].reshape(28, 28)

label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

labels = label_names[y_test[idx]]

_, cache_testing = forward_pass(X_test_reshaped, params)
predicted_label = np.argmax(cache_testing['A4'][idx])
print(f"Predicted Label: {predicted_label}")

plt.imshow(sample_img, cmap='gray')
plt.title(f"True Label: {labels}, Predicted Label: {label_names[predicted_label]}")

accuracy = test_accuracy
plt.xlabel(f"Test Accuracy: {accuracy:.2f}%")
plt.ylabel("Fashion MNIST Sample Image")
plt.show()
