#one hot encoding the labels
def one_hot_encodation(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

y_train_encoded = one_hot_encodation(y_train)
y_test_encoded = one_hot_encodation(y_test)

print(y_train_encoded[0])


#explaining the function
# For each label in y, we create a zero vector of length num_classes (10 for Fashion-MNIST).
# We then set the index corresponding to the label to 1.

#then why in output we see 0s and 1s only in the array?

#  because we are creating a one-hot encoded representation of the labels. and 0's means absence of that class and 1 means presence of that class.

#as we see below in the output its a seqnuence of 0s at first which means absence of that class and 1 at index 9 which means presence of that class

