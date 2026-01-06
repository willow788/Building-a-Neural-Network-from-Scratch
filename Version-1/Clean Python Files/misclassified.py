#plotting a misclassified images
oops_mistakes = np.where(test_predictions != np.argmax(y_test_reshaped, axis=1))[0]
num_mistakes_to_show = 5
plt.figure(figsize=(12, 6))
plt.suptitle('Misclassified Images', fontsize=16)
for i in range(num_mistakes_to_show):
    indx = oops_mistakes[i]
    img = X_test_norm[indx].reshape(28, 28)
    true_label = label_names[y_test[indx]]
    predicted_label = label_names[test_predictions[indx]]
    plt.subplot(1, num_mistakes_to_show, i+1)
    plt.imshow(img, cmap='plasma')
    plt.title(f'True: {true_label}\nPred: {predicted_label}')
    plt.axis('off')
plt.show()
