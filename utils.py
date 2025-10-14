from model import NeuralNetwork

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import seaborn as sns
import copy


def preprocess_data(df, class_column, normalize=True):
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

  df = df.drop(columns=['class']).reset_index(drop=True)
  df = df.join(class_encoded_df)

  return df


def evaluate_networks(dataset, architectures, alphas, regularizers, epochs, k_folds, test_size=0.3, stratify=False, sort_by = 'F1'):
    results = []

    for layers in tqdm(architectures, desc = "Layer Configurations"):
      for alpha in alphas:
        for regularizer in regularizers:
          model = NeuralNetwork(layers=layers, alpha=alpha, regularizer=regularizer, epochs=epochs, k_folds=k_folds, data=dataset)

          if stratify:
            acc, prec, rec, f1 = model.stratified_metrics()
          else:
            train_x, train_y, test_x, test_y = model.fit(test_size=test_size, shuffle=True)
            acc, prec, rec, f1, _ = model.SGD(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

          results.append([len(layers)-2, str(layers), alpha, round(regularizer, 6), epochs, k_folds, test_size, round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4), stratify])

    columns = ['Num Hidden Layers', 'Architecture', 'Alpha', 'Regularizer', 'Epochs', 'Folds', 'Test Size', 'Accuracy', 'Precision', 'Recall', 'F1', 'Stratified']
    results = pd.DataFrame(results, columns = columns)

    if sort_by in columns:
      results = results.sort_values(by = sort_by, ascending = False).reset_index(drop = True)

    return results


def plot_training_metrics(model, metrics=["loss", "accuracy", "precision", "recall", "f1", "confusion_matrix"]):

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
    Plot number of training instances vs. test loss for a given NeuralNetwork model.

    Parameters:
    - model: an initialized (but not yet trained) NeuralNetwork instance.
    - min_frac: minimum training size fraction (e.g. 0.1)
    - max_frac: maximum training size fraction (e.g. 0.9)
    - steps: number of fractional steps to test (default 9 for 0.1 to 0.9)
    """

    train_sizes = np.linspace(min_frac, max_frac, steps)
    test_losses = []

    for frac in tqdm(train_sizes, desc = "Training Size"):

      model_copy = copy.deepcopy(model)

      train_x, train_y, test_x, test_y = model_copy.fit(test_size = 1 - frac, shuffle = True)
      model_copy.SGD(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
      test_losses.append(model_copy.test_losses[-1])  #last epoch test loss

    plt.figure(figsize=(8, 5))
    plt.plot((train_sizes * len(model.data)), test_losses, marker='o')
    plt.xlabel("Number of Training Instances")
    plt.ylabel("Test Loss")
    plt.title("Training Set Size vs. Test Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




