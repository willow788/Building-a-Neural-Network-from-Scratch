#weight initialisating function
def initialisation_weights(input_dimn, hidden1,hidden2, hidden3, output_dimn):
   
   params = {}
   params['W1'] = np.random.randn(input_dimn, hidden1) * np.sqrt(2.0/input_dimn)
   params['b1'] = np.zeros((1, hidden1))

   params['W2'] = np.random.randn(hidden1, hidden2) * np.sqrt(2.0/hidden1)
   params['b2'] = np.zeros((1, hidden2))

   params['W3'] = np.random.randn(hidden2, hidden3) * np.sqrt(2.0/hidden2)
   params['b3'] = np.zeros((1, hidden3))

   params['W4'] = np.random.randn(hidden3, output_dimn) * np.sqrt(2.0/hidden3)
   params['b4'] = np.zeros((1, output_dimn))

   return params
