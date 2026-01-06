
params = initialisation_weights(784, 128, 64, 32, 10)


#training parameters
epoch = 40
learning_rate = 0.01
batch_size = 64


#training loop
for e in range(epoch):

    #np.random.permutation shuffles the data at each epoch
    permutation = np.random.permutation(X_train_norm.shape[0])

    #X_shuffled and y_shuffled are the shuffled versions of X_train_norm and y_train_encoded respectively

    X_shuffled = X_train_norm[permutation]
    y_shuffled = y_train_encoded[permutation]

    #mini-batch gradient descent on the shuffled data

    #range is from 0 to number of samples in X_train_norm with step size of batch_size

    #so here X_train_norm.shape[0] is 60000 as there are 60000 samples in training data

    #so range(0, 60000, 64) will generate numbers from 0 to 60000 with step size of 64. here it will be a batch of 64 samples at a time

    epoch_correct = 0
    epoch_total = 0

    for i in range(0, X_train_norm.shape[0], batch_size):

        # Extract the current mini-batch of data
        #i: starting index of the batch
        #i+batch_size: ending index of the batch

        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        #reshaping the X_batch to (batch_size, 28*28) from (batch_size, 28, 28, 1)
        #because our model input layer is of size 28*28
        #caching the forward pass output
        #so we need to reshape it

        X_batch = X_batch.reshape(X_batch.shape[0], 28*28)

        A4, cache = forward_pass(X_batch, params)

        #calculating the loss by cross entropy loss function

        loss = cross_entropy_loss(y_batch, A4)


        #calculating the gradients by backward function

        grads = backward_function(X_batch, y_batch, params, cache)

        #calculating the accuracy
        #here np.argmax(y_batch, axis=1) gives the true labels from one hot encoded labels
        #so if predictions match with true labels then its correct prediction

        #thus, we calculate the mean of correct predictions to get accuracy
        

        predictions = predict(X_batch, params)
        batch_correct = np.sum(predictions == np.argmax(y_batch, axis=1))
        params = update_weights(params, grads, learning_rate)

        epoch_correct += batch_correct
        epoch_total += X_batch.shape[0]


    epoch_accuracy = epoch_correct / epoch_total
    print(f"Epoch {e+1}/{epoch}, Accuracy: {epoch_accuracy:.4f}")
    print("---------------------------------------------------")
    print("epoch accuracy in percentage :" + str(epoch_accuracy * 100) + "%")




    
