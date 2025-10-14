import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder


class NeuralNetwork:

  def __init__(self, layers, alpha, regularizer, epochs, k_folds, data):
    """Initialize a neural network.

    Parameters:
    - layers: list of ints representing number of neurons in each layer.
    - alpha: learning rate.
    - regularizer: L2 regularization parameter.
    - epochs: number of training epochs.
    - k_folds: number of folds for cross-validation.
    - data: pandas DataFrame with training features and one-hot encoded labels.
    """

    self.layers = layers
    self.alpha = alpha
    self.regularizer = regularizer
    self.epochs = epochs
    self.k_folds = k_folds
    self.data = data
    self.n_layers = len(self.layers)
    self.X = self.data.iloc[:,:-layers[-1]]
    self.Y = self.data.iloc[:,-layers[-1]:]
    self.J = 0
    self.weights = []

    self.training_losses = []
    self.test_losses = []
    self.test_accuracies = []
    self.test_precisions = []
    self.test_recalls = []
    self.test_f1s = []
    self.final_confusion_matrix = None

    self.initialize_weights()


  def initialize_weights(self):
    """Randomly initialize weights for all layers, including bias weights.

    Returns:
    - List of numpy arrays, one for each layer's weights.
    """

    for layer in range(self.n_layers-2):
      self.weights.append(np.random.randn(self.layers[layer + 1], self.layers[layer] + 1))
    self.weights.append(np.random.randn(self.layers[self.n_layers-1], self.layers[self.n_layers-2] + 1))

    return self.weights


  def sigmoid(self, x):
    """Sigmoid activation function."""

    return 1 / (1 + np.exp(-x))


  def forward(self, X):
    """Performs forward propagation.

    Parameters:
    - X: input feature vector.

    Returns:
    - Z: list of pre-activation values.
    - A: list of activations.
    - last_activation: final output vector.
    """

    x = X.copy()
    A = []
    Z = []

    current_activation = np.concatenate(([1], x))
    A.append(current_activation)

    for i in range(len(self.weights)-1):

      z = np.dot(self.weights[i], current_activation)
      Z.append(z)

      current_activation = self.sigmoid(z)
      current_activation = np.concatenate(([1.0], current_activation))

      A.append(current_activation)

    last_layer_z = np.dot(self.weights[-1], A[-1])
    Z.append(last_layer_z)

    last_activation = self.sigmoid(last_layer_z)
    A.append(last_activation)

    return Z, A, last_activation


  def calculate_loss(self, outputs, y):
    """Computes binary cross-entropy loss."""

    loss = -1 * np.multiply(y, np.log(outputs)) - np.multiply((1 - y), np.log(1 - outputs)) #this might run into problems when y is multiple classes
    return np.sum(loss)


  def back_prop(self, X, Y):
    """Performs backpropagation and update weights.

    Parameters:
    - X: input features.
    - Y: true labels.

    Returns:
    - error: list of error vectors.
    - D: list of unregularized gradient matrices.
    - regularized_gradients: list of regularized gradient matrices.
    """

    x = X.copy()#.to_numpy()
    y = Y.copy()#.to_numpy()

    error = []
    D = []
    regularized_gradients = []

    Z, A, output = self.forward(x)

    output_error = np.subtract(output, y)
    error.append(output_error)

    self.J += np.sum(self.calculate_loss(output, y)) #adds the loss for the current training instance to the J class variable

    for layer in range(self.n_layers - 2, -1, -1):

      layer_error_with_bias = np.dot(self.weights[layer].T, error[-1]) * A[layer] * (1 - A[layer])
      layer_error = np.delete(layer_error_with_bias, 0)

      gradient_matrix = np.dot(error[-1].reshape((-1, 1)), A[layer].reshape((1, -1)))
      error.append(layer_error)

      regularized_weights = self.regularizer * self.weights[layer]  #P matrix
      regularized_weights[:,0] = 0    #sets first column to 0

      D.append(gradient_matrix)

      P = np.add(gradient_matrix, regularized_weights) #P is the regularized gradient matrix, #adds D + P(regularized weights)
      regularized_gradients.append(P)

      self.weights[layer] = np.subtract(self.weights[layer], self.alpha * P) #gradient descent on the weights

    return error[:-1], D, regularized_gradients



  def regularize_loss(self, weight_list, loss, n_training_instances, regularizer):

    weights_copy = weight_list.copy()

    S = 0

    for weight_matrix in weights_copy:

      weight_matrix[:,0] = 0      #first column of every weight matrix is bias, so set it to 0
      S = np.sum(np.square(weight_matrix)) #sum of squared weights

    regularized_loss = loss / n_training_instances + regularizer / (2 * n_training_instances) * S
    return regularized_loss


  def fit(self, test_size = 0.3, shuffle = True):

    X = self.X
    Y = self.Y

    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = test_size, shuffle = shuffle)

    return train_x, train_y, test_x, test_y


  def SGD(self, train_x, train_y, test_x = None, test_y = None):

    self.training_losses = []
    self.test_losses = []
    self.test_accuracies = []
    self.test_precisions = []
    self.test_recalls = []
    self.test_f1s = []
    self.confusion_matrix = np.zeros((self.layers[-1], self.layers[-1]))

    x_train_np = train_x.to_numpy()
    y_train_np = train_y.to_numpy()

    n_training_instances = x_train_np.shape[0]

    for i in range(self.epochs):

      self.J = 0 #resets the loss every epoch

      for i in range(n_training_instances):

        x = x_train_np[i]
        y = y_train_np[i]
        instance_loss = self.back_prop(x, y)

      epoch_loss = self.regularize_loss(self.weights, self.J, n_training_instances, self.regularizer)
      self.training_losses.append(epoch_loss)

      if test_x is not None and test_y is not None:
        test_loss, accuracy, precision, recall, f1, confusion_matrix = self.evaluate_on_test_set(test_x, test_y)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(accuracy)
        self.test_precisions.append(precision)
        self.test_recalls.append(recall)
        self.test_f1s.append(f1)

    self.final_confusion_matrix = confusion_matrix
    final_metrics = (self.test_accuracies[-1], self.test_precisions[-1], self.test_recalls[-1], self.test_f1s[-1], self.test_losses[-1])

    return final_metrics


  def predict(self, test_instance):

    _, _, output = self.forward(test_instance)

    prediction = np.zeros(len(output))
    prediction[np.argmax(output)] = 1

    return output, prediction


  def confusion_matrix_metrics(self, cm): #calculates precision, recall, F1 score from the confusion matrix

    num_classes = cm.shape[0]
    eps = 1e-8  #small value to avoid division by zero

    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


  def evaluate_on_test_set(self, test_x, test_y):

    test_x_np = test_x.to_numpy()
    test_y_np = test_y.to_numpy()

    n_test = test_x_np.shape[0]
    correct = 0
    test_loss = 0
    confusion_matrix = np.zeros((self.layers[-1], self.layers[-1]))

    for i in range(n_test):
      output, prediction = self.predict(test_x_np[i])
      actual = test_y_np[i]

      test_loss += np.sum(self.calculate_loss(output, actual))

      actual_index = np.argmax(actual)
      confusion_matrix[actual_index] += prediction

      if np.array_equal(prediction, actual):
          correct += 1

    accuracy = correct / n_test
    precision, recall, f1 = self.confusion_matrix_metrics(confusion_matrix)

    return test_loss, accuracy, precision, recall, f1, confusion_matrix


  def stratify_data(self, k):

    df = self.data.copy()
    n = df.shape[0]

    n_outputs = self.layers[-1]
    folds = [pd.DataFrame() for _ in range(k)]

    y = df.iloc[:,-n_outputs:].idxmax(axis=1)
    x = df.iloc[:,:-n_outputs]
    a = x.join(y.rename('class'))

    classes = pd.unique(a['class'])
    class_dfs = [a[a['class'] == c] for c in classes]

    for class_df in range(n_outputs):

      new = np.array_split(class_dfs[class_df], k)

      for fold in range(k):
        folds[fold] = pd.concat([folds[fold], new[fold]]).sample(frac=1)

    return folds


  def re_hot_encode_class_column(self, df):  #not efficient at all but it works

    n_outputs = self.layers[-1]
    ohe = OneHotEncoder()

    transformed_class = ohe.fit_transform(df[['class']]).toarray()
    temp_df = pd.DataFrame(data = transformed_class).reset_index(drop = True)
    transformed_df = df.drop(columns = ['class']).reset_index(drop = True).join(temp_df, how = 'left', lsuffix='left', rsuffix='right')

    return transformed_df


  def stratified_metrics(self):

    fold_data = self.stratify_data(self.k_folds)
    n_outputs = self.layers[-1]

    accuracy_list = []
    precision_list = []
    recall_list = []
    F1_list = []

    for i in range(len(fold_data)):
      testing_fold = self.re_hot_encode_class_column(fold_data[i])
      k_minus1_fold_data = fold_data[:i] + fold_data[i+1:]

      training_data = self.re_hot_encode_class_column(pd.concat(k_minus1_fold_data))

      train_x = training_data.iloc[:,:-n_outputs]
      train_y = training_data.iloc[:,-n_outputs:]
      test_x = testing_fold.iloc[:,:-n_outputs]
      test_y = testing_fold.iloc[:,-n_outputs:]

      #RE INITIALIZE WEIGHTS AFTER EVERY FOLD

      accuracy, precision, recall, F1, _ = self.SGD(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

      self.weights = []
      self.initialize_weights()

      accuracy_list.append(accuracy)
      precision_list.append(precision)
      recall_list.append(recall)
      F1_list.append(F1)

    return np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(F1_list)




















