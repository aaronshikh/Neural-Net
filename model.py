import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder


class NeuralNetwork:
  '''
  This class implements the full training/evaluation pipeline:
  weight initialization (with explicit bias terms), forward propagation,
  cross-entropy loss, backpropagation with L2 weight decay (bias excluded),
  per-epoch metric logging, and (optional) custom stratified K-fold Cross Validation.

  Attributes
  ----------
  layers : list[int]
      Layer sizes
  alpha : float
      Learning rate used for gradient descent.
  regularizer : float
      L2 penalty coefficient λ (bias weights are not penalized).
  epochs : int
      Number of passes over the training data.
  k_folds : int
      Number of folds for stratified cross-validation.
  data : pandas.DataFrame
      Preprocessed dataset where the last layers[-1] columns are one-hot labels.
  X : pandas.DataFrame
      Feature matrix extracted from data.
  Y : pandas.DataFrame
      One-hot label matrix extracted from data.
  weights : list[np.ndarray]
      One weight matrix per layer (including a bias column in each).
      Shape of W^i is (n_{i+1}, n_{i} + 1). Where W^i is the weight matrix for layer i (connecting layer i -> layer i + 1)
  training_losses, test_losses : list[float]
      Regularized losses per epoch.
  test_accuracies, test_precisions, test_recalls, test_f1s : list[float]
      Evaluation metrics (when a test set is provided) per epoch.
  final_confusion_matrix : np.ndarray
      Confusion matrix computed on the latest evaluation pass.
  '''

  def __init__(self, layers, alpha, regularizer, epochs, k_folds, data):

    """
    Initializes the network configuration and training state.

    Parameters
    ----------
    layers : list[int]
        Layer sizes from input to output (e.g., [48, 12, 4, 2]).
        First value must equal the number of input features.
        The last value must equal the number of one-hot label columns in data.
    alpha : float
        Learning rate for weight updates.
    regularizer : float
        L2 penalty coefficient λ (bias weights are not penalized).
    epochs : int
        Number of training epochs to run.
    k_folds : int
        Number of folds for the custom stratified cross-validation routine.
    data : pandas.DataFrame
        Preprocessed dataset where the last layers[-1] columns are one-hot encoded labels.

    Notes
    -----
    - Splits data into features (self.X) and labels (self.Y).
    - Initializes network weights via initialize_weights().
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
    """
    Populates self.weights as a list[np.ndarray] with one matrix per layer. 
    
    Each weight matrix includes an extra first column for the bias term.
    For a pair of consecutive layers (n_in -> n_out), the shape is
    (n_out, n_in + 1). Weights are drawn from a standard normal distribution.
    """

    for layer in range(self.n_layers-2):
      self.weights.append(np.random.randn(self.layers[layer + 1], self.layers[layer] + 1))
    self.weights.append(np.random.randn(self.layers[self.n_layers-1], self.layers[self.n_layers-2] + 1))

    return self.weights


  def sigmoid(self, x):
    """Sigmoid activation function."""

    return 1 / (1 + np.exp(-x))


  def forward(self, X):
    """
    Performs forward propagation of a single training or test instance through all layers using sigmoid activations.

    Parameters
    ----------
    X : np.ndarray | pandas.DataFrame
        Feature vector of shape (n_features, 1).

    Returns
    -------
    Z_list : list[np.ndarray]
        Linear pre-activations per layer (including bias concatenation step in A), for debugging.
    A_list : list[np.ndarray]
        Activations per layer (with leading 1.0 inserted to carry the bias).
    output : np.ndarray
        The final layer activation (sigmoid probabilities) of shape (n_outputs, 1).
        Last entry of A_list.
        Argmaxing this would give the class prediction.

    Notes
    -----
    - Bias handling: a 1 is prepended to each layer’s activation
      so the bias weights are multiplied automatically on the next layer’s dot product.
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
    """
    Computes (unregularized) cross-entropy loss.

    Parameters
    ----------
    outputs : np.ndarray
        Sigmoid probabilities from the final layer calculated from forward(); shape: (n_outputs, 1).
    y : np.ndarray
        One-hot-encoded true labels; shape: (n_outputs, 1).

    Returns
    -------
    float
        Sum of element-wise cross-entropy over the output list for a single instance.
    """

    loss = -1 * np.multiply(y, np.log(outputs)) - np.multiply((1 - y), np.log(1 - outputs))
    return np.sum(loss)
  

  def regularize_loss(self, weight_list, loss, n_training_instances, regularizer):
    """
    Add L2 penalty to the loss (bias excluded).

    Parameters
    ----------
    weight_list : list[np.ndarray]
        Current weight matrices for all layers (self.weights, each includes a bias column).
    loss : float
        Unregularized loss value (sum, self.J, of per-sample cross entropies calculated from calculate_loss()).
    n_training_instances : int
        Sample size (m).
    regularizer : float
        L2 coefficient λ.

    Returns
    -------
    float
        Regularized loss: loss / m + (λ / (2m)) * Σ ||W_no_bias||^2

    Notes
    -----
    - Bias weights (first column) are excluded from the penalty.
    """

    weights_copy = weight_list.copy()

    S = 0

    for weight_matrix in weights_copy:

      weight_matrix[:,0] = 0      #first column of every weight matrix is bias, so set it to 0
      S += np.sum(np.square(weight_matrix)) #sum of squared weights

    regularized_loss = loss / n_training_instances + regularizer / (2 * n_training_instances) * S
    return regularized_loss


  def back_prop(self, X, Y):
    """
    Perform backpropagation and update the network’s weights using stochastic gradient descent.

    This method computes layer wise errors via the chain rule, calculates both raw and
    regularized gradients, and applies a weight update step for each layer.

    Parameters
    ----------
    X : np.ndarray | pandas.DataFrame
        Input feature vector of shape: (n_features, 1).
    Y : np.ndarray | pandas.DataFrame
        One-hot-encoded labels of shape: (n_outputs, 1).

    Returns (For debugging or future use)
    -------
    error : list[np.ndarray]
        Backpropagated error terms (δ) for each layer.
        Each element corresponds to the layer’s partial derivative of loss w.r.t. z.
    D : list[np.ndarray]
        Unregularized gradient matrices (∂J/∂W) for each layer.
    regularized_gradients : list[np.ndarray]
        Gradients including L2 regularization (D + λW), where the bias column is excluded of each layer.

    Notes
    -----
    - Performs a full forward pass to obtain activations and predictions using forward().
    - Uses the sigmoid activation derivative (A * (1 - A)) for hidden layers.
    - Bias terms are excluded from regularization by zeroing their corresponding weights.
    - Weight update rule:
          W := W - α * (D + λW)   where α is the learning rate (self.alpha).
    - The cumulative training loss is stored in self.J.
    - This function effectively trains the network one training instance at a time.
    """

    x = X.copy()
    y = Y.copy()

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

      P = np.add(gradient_matrix, regularized_weights) #adds D + P(regularized weights), where P is the regularized gradient matrix
      regularized_gradients.append(P)

      self.weights[layer] = np.subtract(self.weights[layer], self.alpha * P) #gradient descent on the weights

    return error[:-1], D, regularized_gradients


  def fit(self, test_size = 0.3, shuffle = True):
    """
    This is a simple wrapper around sklearn.model_selection.train_test_split, returning
    DataFrames ready for training.
    """

    X = self.X
    Y = self.Y

    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X, Y, test_size = test_size, shuffle = shuffle)

    return train_x, train_y, test_x, test_y


  def train(self, train_x, train_y, test_x = None, test_y = None):
    """
    Trains the neural network using stochastic gradient descent.

    This method iterates over the training data for a fixed number of epochs (self.epochs),
    performing forward and backward propagation on each instance, updating weights
    using L2-regularized gradients. After each epoch, it logs the training loss
    and computes and records test set performance metrics.

    Parameters
    ----------
    train_x : pandas.DataFrame
        Training feature matrix of shape (training_set_size, n_features). Where m is the size of the training set
    train_y : pandas.DataFrame
        One-hot encoded training labels of shape (training_set_size, n_outputs).
    test_x : pandas.DataFrame, optional
        Test feature matrix for evaluation of shape (test_set_size, n_features).
    test_y : pandas.DataFrame, optional
        One-hot-encoded test labels for evaluation of shape (test_set_size, n_outputs).

    Returns
    -------
    final_metrics : tuple[float, float, float, float, float]
        The final test metrics after the last epoch in the form:
        (accuracy, precision, recall, F1 score, test loss).

    Notes
    -----
    - Performs a full training loop:
        1. Resets cumulative loss self.J each epoch.
        2. Performs back_prop() for every training instance.
        3. Computes the regularized epoch loss and appends it to self.training_losses.
        4. Evaluates the model via evaluate_on_test_set() function
           to log loss, accuracy, precision, recall, and F1 score per epoch.
    - The final epoch’s confusion matrix is saved in self.final_confusion_matrix.
    - Weight updates use the learning rate self.alpha and regularization self.regularizer using stochastic gradient descent.
    - Metrics arrays can later be visualized using utils.plot_training_metrics().
    """

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
    """
    Run a single forward pass and return both probabilities and a one-hot-encoded class prediction.

    Parameters
    ----------
    test_instance : np.ndarray
        A single input feature vector of length n_features.

    Returns
    -------
    output : np.ndarray
        The network's output probabilities for each class of shape: (n_outputs, 1).
    prediction : np.ndarray
        One-hot-encoded class vector of shape (n_outputs, 1) with 1 at argmax(output) and 0 elsewhere.
    """

    _, _, output = self.forward(test_instance)

    prediction = np.zeros(len(output))
    prediction[np.argmax(output)] = 1

    return output, prediction


  def confusion_matrix_metrics(self, cm): #calculates precision, recall, F1 score from the confusion matrix
    """
    Compute macro-averaged precision, recall, and F1 from a confusion matrix.

    Parameters
    ----------
    cm : list[np.ndarray]
        Confusion matrix of shape (n_classes, n_classes), where rows correspond to
        actual classes and columns to predicted classes.

    Returns
    -------
    precision : float
        Macro-averaged precision across classes.
    recall : float
        Macro-averaged recall across classes.
    f1 : float
        Macro-averaged F1 score across classes.

    Notes
    -----
    - Uses a small epsilon to avoid division by zero.
    - Per-class metrics: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
      are computed and used to calculate precision, recall, f1.
    """

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
    """
    Helper fnction to evaluate loss and classification metrics on a test set after weights have been trained.

    Parameters
    ----------
    test_x : pandas.DataFrame
        Test features of shape (test_set_size, n_features).
    test_y : pandas.DataFrame
        One-hot-encoded test labels of shape (test_set_size, n_outputs).

    Returns
    -------
    test_loss : float
        Sum of (unregularized) cross-entropy losses over the test set.
    accuracy : float
        Overall accuracy = correct / test_set_size.
    precision : float
        Macro-averaged precision from the confusion matrix.
    recall : float
        Macro-averaged recall from the confusion matrix.
    f1 : float
        Macro-averaged F1 from the confusion matrix.
    confusion_matrix : list[np.ndarray]
        Integer confusion matrix aggregated over the test set with shape: (n_outputs, n_outputs).
        Used for visualizations later.

    Notes
    -----
    - Iterates through the test set and calls predict(), accumulates loss with
      calculate_loss(), and fills the confusion matrix.
    - Calls confusion_matrix_metrics() to get the recall, precision, and f1
    """

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
    """
    Create k stratified folds that preserve class proportions.

    Parameters
    ----------
    k : int
        Number of folds to create.

    Returns
    -------
    folds : list[pandas.DataFrame]
        List of k dataframes, each containing features and a single 'class' column
        (not one-hot-encoded). Class distribution is approximately preserved in each fold.

    Notes
    -----
    - One-hot label columns are reintroduced later by re_hot_encode_class_column().
    """

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


  def re_hot_encode_class_column(self, df):
    """
    Helper function for stratify_data() that replaces a single categorical 'class' column with one-hot-encoded label columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Fold dataframe containing features + a single 'class' column.

    Returns
    -------
    transformed_df : pandas.DataFrame
        The same dataframe with 'class' removed and replaced by one-hot columns
        appended at the end (number of columns equals n_outputs).

    Notes
    -----
    - Probably not the most efficient method and may be a little clumsy, but it works
    """

    ohe = OneHotEncoder()

    transformed_class = ohe.fit_transform(df[['class']]).toarray()
    temp_df = pd.DataFrame(data = transformed_class).reset_index(drop = True)
    transformed_df = df.drop(columns = ['class']).reset_index(drop = True).join(temp_df, how = 'left', lsuffix = 'left', rsuffix = 'right')

    return transformed_df


  def stratified_metrics(self):
    """
    Run custom stratified K-fold cross-validation and return mean metrics.

    Process
    -------
    1) Build k stratified folds via stratify_data(self.k_folds).
    2) For each fold i:
       - Use fold i as test; concatenate the remaining folds as train.
       - Split training and test sets into features and outputs (X, Y).
       - Train/evaluate with train()) on that split to obtain metrics.
       - Re-initialize weights after each fold to avoid leakage between folds.
    3) Average accuracy, precision, recall, and F1 across folds.

    Returns
    -------
    accuracy_mean : float
    precision_mean : float
    recall_mean : float
    f1_mean : float
    """

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

      accuracy, precision, recall, F1, _ = self.train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

      self.weights = []
      self.initialize_weights()

      accuracy_list.append(accuracy)
      precision_list.append(precision)
      recall_list.append(recall)
      F1_list.append(F1)

    return np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(F1_list)








