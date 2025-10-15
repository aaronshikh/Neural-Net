# Neural Network From Scratch

This project implements a fully connected feedforward neural network from scratch using only NumPy and no TensorFlow, PyTorch, or scikit-learn model wrappers.  Its intention is to demonstrate a full understanding of basic neural networks: forward propagation, backpropagation, gradient descent, regularization, and model evaluation.

---

## Implementation Details

### `model.py`:

This file contains the full implementation of a feedforward neural network using only **NumPy** and **pandas**, so no deep learning libraries. It shows the mathematical and algorithmic foundations of backpropagation, regularization, and gradient descent.

#### NeuralNetwork Class
The `NeuralNetwork` class encapsulates all components of a traditional supervised neural network pipeline:

- **Initialization**: Sets up model architecture (`layers`), learning rate (`alpha`), L2 regularization coefficient (`regularizer`), epochs, and number of folds for cross-validation. The input and target matrices are parsed automatically from the dataset.

   The argument **`layers`** is a list defining the size of each layer in the network.  
    Example:  
    ```python
    layers = [48, 12, 2]
    ```  
    means:
    - 48 input features  
    - 1 hidden layer with 12 neurons  
    - 2 output neurons (for binary classification)
    - 
  This flexible list structure allows you to define arbitrary network depths and widths for each dataset.

- **Weight Initialization (`initialize_weights`)**: Randomly initializes layer weights with an added bias term for each layer.
- **Forward Propagation (`forward(X)`)**: Computes activations layer by layer using the **sigmoid** function. Biases are prepended to each layer’s activation vector, and intermediate activations/pre-activations are cached for gradient computation.

