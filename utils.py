from model import NeuralNetwork

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import seaborn as sns
import copy


def preprocess_data(df, class_column, normalize = True):
  """
  Prepare a dataset for neural network training by normalizing features
  and one-hot-encoding the class labels.

  Parameters
  ----------
  df : pandas.DataFrame
      Input dataset containing feature columns and a target column.
  class_column : str
      Name of the target column to be moved and one-hot-encoded.
  normalize : bool, optional (default=True)
      Whether to normalize feature columns to the range [0, 1].

  Returns
  -------
  df_processed : pandas.DataFrame
      DataFrame with normalized features and one-hot encoded class columns appended.

  Notes
  -----
  - Uses sklearn’s OneHotEncoder() to encode the class column into multiple binary columns.
  - Output now format matches the expected input structure for the NeuralNetwork class, where the
    last layers[-1] columns represent the target labels.
  """

  df = df.copy()

  class_col = df.pop(class_column)
  df['class'] = class_col

  if normalize:
    for col in df.columns:
      if col != 'class':
        max_val = df[col].max()
        min_val = df[col].min()
        df[col] = (df[col] - min_val) / (max_val - min_val)

  ohe = OneHotEncoder()
  class_encoded = ohe.fit_transform(df[['class']]).toarray()
  class_encoded_df = pd.DataFrame(class_encoded)

  df_processed = df.drop(columns=['class']).reset_index(drop=True)
  df_processed = df.join(class_encoded_df)

  return df_processed


def evaluate_networks(dataset, architectures, alphas, regularizers, epochs, k_folds, test_size=0.3, stratify=False, sort_by = 'F1'):
  """
  Trains and evaluates multiple neural network configurations and returns a summary table. Esentially a hyperparameter grid search.

  Parameters
  ----------
  dataset : pandas.DataFrame
      Preprocessed dataset ready for training (features + one-hot-encoded labels).
  architectures : list[list[int]]
      List of layer configurations (e.g. [[48, 8, 2], [48, 16, 8, 2]]).
  alphas : list[float]
      Learning rates to test.
  regularizers : list[float]
      L2 regularization coefficients (λ values) to test.
  epochs : int
      Number of epochs for each training run.
  k_folds : int
      Number of folds for stratified cross-validation (if enabled).
  test_size : float, optional (default=0.3)
      Fraction of data reserved for testing when not using stratified cross validation.
  stratify : bool, optional (default=False)
      Whether to use the model’s custom stratified cross-validation method.
  sort_by : str, optional (default='F1')
      Metric name to sort the results by.

  Returns
  -------
  results : pandas.DataFrame
      Summary of all tested configurations including architecture, alpha, λ,
      epochs, folds, test size, and performance metrics (Accuracy, Precision,
      Recall, F1). Sorted descending by the selected metric.

  Notes
  -----
  - Iterates over all combinations of layers, alpha, and λ.
  - If stratify=True, uses model.stratified_metrics(); otherwise performs a
    simple train/test split via model.fit() and model.train().
  """
  results = []

  for layers in tqdm(architectures, desc = "Layer Configurations"):
    for alpha in alphas:
      for regularizer in regularizers:
        model = NeuralNetwork(layers=layers, alpha=alpha, regularizer=regularizer, epochs=epochs, k_folds=k_folds, data=dataset)

        if stratify:
          acc, prec, rec, f1 = model.stratified_metrics()
        else:
          train_x, train_y, test_x, test_y = model.fit(test_size=test_size, shuffle=True)
          acc, prec, rec, f1, _ = model.train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

        results.append([len(layers)-2, str(layers), alpha, round(regularizer, 6), epochs, k_folds, test_size, round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4), stratify])

  columns = ['Num Hidden Layers', 'Architecture', 'Alpha', 'Regularizer', 'Epochs', 'Folds', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1', 'Stratified']
  results = pd.DataFrame(results, columns = columns)

  if sort_by in columns:
    results = results.sort_values(by = sort_by, ascending = False).reset_index(drop = True)

  return results


def plot_training_metrics(model, metrics=["loss", "accuracy", "precision", "recall", "f1", "confusion_matrix"]):
  """
    Visualizes training and testing performance metrics over epochs for a trained NeuralNetwork model.

    Parameters
    ----------
    model : NeuralNetwork
        Trained neural network instance from model.py containing recorded metric histories.
    metrics : list[str], optional
        Which metrics to plot. Options: 'loss', 'accuracy', 'precision', 'recall',
        'f1', 'confusion_matrix'. Defaults to all.

    Notes
    -----
    - Generates one subplot per metric (except confusion matrix).
    - If 'confusion_matrix' is included, renders a heatmap of the final confusion matrix.
    """

  epochs = range(1, model.epochs + 1)
  plot_metrics = [m for m in metrics if m != "confusion_matrix"]

  fig, axs = plt.subplots(1, len(plot_metrics), figsize=(6 * len(plot_metrics), 5))
  axs = np.atleast_1d(axs)

  for ax, metric in zip(axs, plot_metrics):
    if metric == "loss":
      ax.plot(epochs, model.training_losses, label="Training Loss")
      ax.plot(epochs, model.test_losses, label="Test Loss")
      ax.set_title("Loss over Epochs")
    elif metric == "accuracy":
      ax.plot(epochs, model.test_accuracies, label="Accuracy")
      ax.set_title("Test Accuracy over Epochs")
    elif metric == "precision":
      ax.plot(epochs, model.test_precisions, label="Precision")
      ax.set_title("Test Precision over Epochs")
    elif metric == "recall":
      ax.plot(epochs, model.test_recalls, label="Recall")
      ax.set_title("Test Recall over Epochs")
    elif metric == "f1":
      ax.plot(epochs, model.test_f1s, label="F1 Score")
      ax.set_title("Test F1 Score over Epochs")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    ax.grid(True)

  if "confusion_matrix" in metrics:
    plt.figure(figsize=(6, 5))
    sns.heatmap(model.final_confusion_matrix, annot = True, fmt=".1f", cmap = "Blues", cbar = False)
    plt.title("Final Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

  plt.tight_layout()
  plt.show()


def plot_learning_curve(model, min_frac=0.1, max_frac=0.9, steps=9):
    """
    Plot a learning curve showing how test loss changes with incremental training set size.

    Parameters
    ----------
    model : NeuralNetwork
        Initialized (untrained) neural network instance to be cloned and trained repeatedly.
    min_frac : float, optional (default=0.1)
        Minimum fraction of the dataset to use for training.
    max_frac : float, optional (default=0.9)
        Maximum fraction of the dataset to use for training.
    steps : int, optional (default=9)
        Number of fractional steps to evaluate between min_frac and max_frac.

    Notes
    -----
    - For each fraction, deep-copies the model and trains it with the corresponding
      subset size (using train() internally).
    - Records the final test loss from each training run.
    - Plots test loss versus number of training instances to show how model
      performance scales with data size.
    """

    train_sizes = np.linspace(min_frac, max_frac, steps)
    test_losses = []

    for frac in tqdm(train_sizes, desc = "Training Size"):

      model_copy = copy.deepcopy(model)

      train_x, train_y, test_x, test_y = model_copy.fit(test_size = 1 - frac, shuffle = True)
      model_copy.train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
      test_losses.append(model_copy.test_losses[-1])  #last epoch test loss

    plt.figure(figsize=(8, 5))
    plt.plot((train_sizes * len(model.data)), test_losses, marker='o')
    plt.xlabel("Number of Training Instances")
    plt.ylabel("Test Loss")
    plt.title("Training Set Size vs. Test Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




