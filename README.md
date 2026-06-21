# Neural Network from Scratch — Optimisation in Machine Learning

From-scratch implementation of gradient descent and a fully connected neural network with backpropagation and mini-batch stochastic gradient descent, applied to a 2D function regression task.

## Context

MSc in Statistics & Data Science, University College Dublin — Optimisation in Machine Learning module. The assignment builds two models entirely from first principles using NumPy, without any ML framework, to demonstrate understanding of the mathematical foundations underlying modern AI systems.

## What it does

### Task 1 — Polynomial Regression

- Constructs a degree-4 polynomial design matrix (15 basis functions) over a 2D input space.
- **Method 1:** Gradient descent with backtracking line search — steepest-descent direction, Armijo sufficient-decrease condition, convergence to tolerance 1e-8.
- **Method 2:** Convex optimisation solver (`cvxpy`) as a ground-truth reference.
- Both methods are compared analytically; contour plots of the absolute fractional error |model − data| / |data| over the input domain.

### Task 2 — Neural Network

- Implements a feedforward neural network class from scratch: arbitrary depth and width, sigmoid hidden activations, linear output layer.
- **Backpropagation** derived and coded layer by layer using the chain rule.
- **Mini-batch SGD** with configurable learning rate, epochs and batch size.
- **Architecture search** across 7 topologies (e.g. `[2,3,1]`, `[2,5,1]`, `[2,2,2,1]`) with 10 random initialisations each; best topology selected by MSE on training data.

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Numerical computing | `numpy` |
| Optimisation | `cvxpy` |
| Visualisation | `matplotlib` |

## Key results

- Gradient descent and cvxpy converge to the same solution (max parameter difference < 1e-6), confirming correct implementation.
- Best neural network topology: **[2, 5, 1]** (21 parameters); trained for 5,000 epochs with mini-batch size 3 and learning rate 0.15.
- The neural network captures non-linear structure in the 2D input space that the polynomial regression cannot represent in high-residual regions.

## Data

The training dataset (`25221241_training_data.dat`) was provided by UCD and is not redistributed here. Place it in `data/` before running `src/neural_network.py`.
