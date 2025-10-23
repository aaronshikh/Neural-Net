# Neural Network From Scratch

This project implements a fully connected feedforward neural network from scratch using only NumPy and no TensorFlow, PyTorch, or scikit-learn model wrappers. It intends to demonstrate a full understanding of basic neural networks: forward propagation, backpropagation, gradient descent, regularization, and model evaluation on three basic datasets.

---


## Table of Contents
- [Implementation Details](#implementation-details)
  - [`model.py`](#modelpy)
  - [`utils.py`](#utilspy)
- [Example Usage](#example-usage)
- [Datasets](#datasets)
- [Evaluation and Results](#evaluation--results)
  - [`results.ipynb`](#resultsipynb)
- [Limitations & Future Work](#limitations--future-work)



## Implementation Details

### `model.py`:

This file contains the full implementation of a feedforward neural network using only **NumPy** and **pandas**, so no deep learning libraries. It shows the mathematical and algorithmic foundations of backpropagation, regularization, and gradient descent.

#### NeuralNetwork Class
The `NeuralNetwork` class encapsulates all components of a traditional supervised neural network pipeline:

- **Initialization**: Sets up model architecture (`layers`), learning rate (`alpha`), L2 regularization coefficient (`regularizer`), epochs, and number of folds for cross-validation. The input and target matrices are parsed automatically from the dataset.

   The argument **`layers`** is a list defining the size of each layer in the network.  
    Example:  
    ```python
    layers = [48, 12, 4, 2]
    ```  
    Means:
    - 48 input features  
    - 2 hidden layers with 12 and 4 neurons  
    - 2 output neurons (for binary classification)
      
  This flexible list structure allows us to define arbitrary network depths and widths for each dataset.

- **Weight Initialization (`initialize_weights()`)**: Allocates a random weight matrix per layer using the normal distribution.
  
- **Forward Propagation (`forward(X)`)**: Takes one input vector, pushes it through the network with sigmoid activations, and returns the final output prediction.
  
- **Backpropagation + Update (`back_prop(x, y)`)**: Takes one training instance and, after forward propogation, propagates the error backward starting from the output layer, multiplying by the derivative of the sigmoid activation (A × (1 − A)) to obtain ∂J/∂W (the weight gradients), and adding L2 regularization. The weights are then updated via stochastic gradient descent using the learning rate (alpha).
  
- **Training Loop (`train(train_x, train_y, test_x, test_y)`)**  
This function orchestrates the full learning process. It repeatedly performs forward and backward passes through the network, updating weights using gradient descent and keeping track of performance metrics after each epoch.

  **Training steps:**
  1. **Epoch loop:** For each epoch:
     - Keeps track of regularized loss.
     - Iterates over all training instances:
       - Runs `forward(x)` to compute outputs and activations.
       - Calls `back_prop(x, y)` to compute gradients and update weights using stochastic gradient descent.
     - Continues to update weights over multiple runs through the training set.
  2. **Testing phase:**  
     - Calls `evaluate_on_test_set()` to compute:
        - Test loss  
        - Accuracy, Precision, Recall, and F1  
        - Confusion matrix  
     - These metrics are logged each epoch to track learning progress.
  3. **Output:**  
     Returns the final test metrics after the last epoch and stores the complete learning curves for visualization.

- **Stratified Cross-Validation (`stratified_metrics()`)**  
This method extends the standard `train()` process by performing **k-fold stratified cross-validation**, ensuring that each fold preserves the original class distribution. Instead of taking predefined training and testing sets, it internally divides the dataset into `k_folds` (defined during initialization) and evaluates the model across all of them.

  **Process:**
  1. **Stratified splitting:**  
     - The dataset is partitioned into `k` folds using `stratify_data()`, which keeps class proportions consistent across folds.
  2. **Fold iteration:**  
     For each fold:
     - The fold is left out as the **test set**, and the remaining `k−1` folds are concatenated as the **training set**.
  3. **Training per fold:**  
     - Calls `train()` internally on the current fold pair (train/test), running the full epoch-by-epoch training cycle.
     - Weights are reinitialized using `initialize_weights()` to prevent information leakage between folds.
  5. **Metric aggregation:**  
     Accuracy, precision, recall, and F1 scores are collected from each fold and averaged to produce the final cross-validation results.

 
### `utils.py`:

This file contains utility functions for **data preprocessing**, **hyperparameter evaluation**, and **model visualization**.  
These functions handle all the data setup and plotting so the neural network logic in `model.py` remains clean and focused.

---

#### Overview
- **Preprocessing:** Normalizes features and one-hot encodes labels in the exact format the model expects.  
- **Evaluation:** Run a grid search over architectures/α/λ/epochs; optionally use stratified Cross Validation.  
- **Visualization:** Plot training/test curves, confusion matrix, and a learning curve (test loss vs. train size).


## Example Usage

```
from model import NeuralNetwork
from utils import preprocess_data, evaluate_networks, plot_training_metrics, plot_learning_curve

# 1) Preprocess
df = preprocess_data(raw_df, class_column = "class", normalize = True)

# 2) Sweep hyperparameters
grid = evaluate_networks(
    dataset = df,
    architectures = [[48,12,2], [48,16,8,2]],
    alphas = [0.01, 0.02, 0.05],
    regularizers = [0.0001, 0.001, 0.01, 0.1],
    epochs = 50,
    k_folds = 10,
    test_size = 0.4,
    stratify = True,
    sort_by = "F1"
)
best = grid.iloc[0]

# 3) Re-train best config and visualize
nn = NeuralNetwork( layers = best["Architecture"],
                    alpha = best["Alpha"],
                    regularizer = best["Regularizer"],
                    epochs = best["Epochs"],
                    k_folds = best["Folds"],
                    data = df
                   )
train_x, train_y, test_x, test_y = nn.fit(test_size = best["Test Size"], shuffle = True)
nn.train(train_x, train_y, test_x, test_y)

plot_training_metrics(nn)
plot_learning_curve(nn)

```


## Datasets

The neural network was evaluated on three classic datasets spanning binary and multiclass classification problems.  
Each dataset was preprocessed using `utils.preprocess_data()` (normalization and one-hot encoding) before training.

Each dataset section in the notebook includes:
- Grid search over hyperparameters (layer architecture, α, λ, epochs)
- 10-fold cross-validation or held-out test split
- Visualizations of training/test loss, performance metrics, and confusion matrices

---

### House Votes 84
- **Task:** Binary classification. Predict **party affiliation** (Democrat vs. Republican).  
- **Description:** Based on the 1984 U.S. Congressional Voting Records dataset. Each sample represents a House member’s votes across 16 key issues, one-hot encoded into **48 binary input features**.  
- **Goal:** Classify members by political party based on their voting patterns.  
- **Observations:** Extremely separable dataset, so even shallow networks achieved almost perfect accuracy and F1.  
  - Best configuration: Layers = `[48, 12, 2]`, α = 0.02, λ = 0.001
  - Final Results: F1 = 0.9823, Accuracy = 0.9828.

---

### Wine Recognition
- **Task:** Multiclass classification. Predict **wine cultivar** (three classes).  
- **Description:** Derived from the UCI Wine dataset. Each sample contains **13 continuous chemical measurements** (e.g., alcohol, malic acid, flavonoids, color intensity).  
- **Goal:** Identify which of the three grape cultivars a wine came from.  
- **Observations:** Required a deeper network and a slightly higher learning rate.  
  - Best configuration: Layers = `[13, 16, 8, 3]`, α = 0.05, λ = 0.001
  - Final Results: F1 = 0.9896, Accuracy = 0.9889.

---

### Breast Cancer
- **Task:** Binary classification. Detect **tumor malignancy** (benign vs. malignant).  
- **Description:** Simplified version of the original UCI Breast Cancer dataset containing **9 numeric cytological features** describing cell shape, adhesion, and nucleus size.  
- **Goal:** Classify each tumor sample as malignant or benign.  
- **Observations:** Small dataset benefited from moderate regularization to prevent overfitting.  
  - Best configuration: Layers = `[9, 12, 6, 2]`, α = 0.05, λ = 0.01
  - Final Results: F1 = 0.9660, Accuracy = 0.9686.

---

## Evaluation & Results

### `results.ipynb`:

Each dataset was evaluated using a grid search over network architectures, learning rates (α), and L2 regularization parameters (λ).  
Performance was measured using **Accuracy**, **Precision**, **Recall**, and **F1 Score**, averaged over epochs and/or cross-validation folds.

### Evaluation Workflow
1. **Grid Search** – The `evaluate_networks()` function iterates through all layer/α/λ combinations, trains a model, and records metrics in a summary DataFrame.
2. **Best Model Selection** – The configuration with the highest F1 score is chosen and retrained on the full training set.
3. **Visualization** – The training and testing progress is visualized using `plot_training_metrics()` (per-epoch metrics) and `plot_learning_curve()` (test loss vs. train size).




