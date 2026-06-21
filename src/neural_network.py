
import numpy as np
import matplotlib.pyplot as plt

##########################################################################################################
# TASK 1. LINEAR REGRESSION
##########################################################################################################
# Load training data
training_data = np.loadtxt('data/training_data.dat')

xs = training_data[:,:2] # inputs
ys = training_data[:,2]  # outputs
n = len(xs[:, 0]) # size of training data

# for plot, rearrange the outputs into a square matrix with axes following the two input dimensions
ys_matrix = ys.reshape(round(np.sqrt(len(ys))),round(np.sqrt(len(ys)))).transpose()

# contour plot with axes representing input values and contours of constant output value
cf = plt.contourf(ys_matrix, extent=[-1,1,-1,1]) # fill levels
contours = plt.contour(ys_matrix, extent=[-1,1,-1,1], colors='k', linewidths=0.5)
plt.title('Contour plot of training data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.clabel(contours)
plt.colorbar(cf, label='y')   # we add cf to colour the bar
plt.axis('scaled')
plt.title('Contour plot of training data') 
plt.tight_layout()
plt.savefig('training_data.png', dpi=150)


x1 = xs[:, 0]   # first input
x2 = xs[:, 1]   # second input
# Build design matrix X  (441(len(x1)) × 15(parameters))
X = np.column_stack([
    np.ones_like(x1),    # c00 : 1
    x1,                  # c10 : x1
    x2,                  # c01 : x2
    x1**2,               # c20 : x1^2
    x2**2,               # c02 : x2^2
    x1*x2,               # c11 : x1*x2
    x1**3,               # c30 : x1^3
    x2**3,               # c03 : x2^3
    x1**2*x2,            # c21 : x1^2*x2
    x1*x2**2,            # c12 : x1*x2^2
    x1**4,               # c40 : x1^4
    x2**4,               # c04 : x2^4
    x1**3*x2,            # c31 : x1^3*x2
    x1*x2**3,            # c13 : x1*x2^3
    x1**2*x2**2,         # c22 : x1^2*x2^2
])


# Method 1: Gradient descent with backtracking line search
###############################################################################

# loss function (AKA cost function)
def loss(theta, X, ys): # vector that minmizes the loss function
    # sum of squares of errors between model and training data
    return 0.5 * np.linalg.norm(X @ theta - ys)**2

# Gradient of the loss function X^T (X theta - y)
def grad_loss(theta, X, ys):
    return X.T @ (X @ theta - ys)

# Backtracking line search
def backtrack_step(grad, theta, X, ys):
    p = -grad / np.linalg.norm(grad)   # unit steepest-descent direction
    alpha = 1.0
    t = 0.5           # reduction factor
    c = 1e-4          # sufficient-decrease constant
    # how much smaller does alpha need to be
    while loss(theta + alpha*p, X, ys) > loss(theta, X, ys) + c*alpha*np.dot(p, grad):
        alpha = t*alpha
        if alpha < 1e-6:
            break
    return alpha*p  # how much does alpha change

# implement gradianet descent to minimize loss function
tol = 1e-8
# initial guess for theta vector
thetas = [np.ones(15)]   # initial guess: all ones

while True:
    grad = grad_loss(thetas[-1], X, ys)
    # each cycle of loop: descend along gradient towards minimum
    dtheta = backtrack_step(grad, thetas[-1], X, ys) # theta change proportional to negative gradient
    # update theta vector
    thetas.append( thetas[-1]+dtheta )
    # when we are close enough to minimum, break
    if np.abs( loss(thetas[-1],X,ys) - loss(thetas[-2],X,ys) ) < tol: 
        break

out_grad = thetas[-1]

# Method 2: cvxpy (convex optimisation solver)
###############################################################################
import cvxpy as cp
# Create a (15) vector of variables
Theta = cp.Variable(15)

# Minimize same sum-of-squares loss function
objective = cp.Minimize(0.5 * cp.sum_squares(X @ Theta - ys))

prob = cp.Problem(objective)
prob.solve()

# Create problem and solve it
out_cvxpy = Theta.value   # numpy array of the 15 optimal coefficients
print(out_cvxpy)


# Evaluate model and fractional error
###############################################################################
y_pred = X @ out_cvxpy 
frac_err = np.abs(y_pred - ys) / np.abs(ys)


# Contour plot: absolute fractional error |model - data| / |data|
###############################################################################
# The training data sits on a 21x21 grid; reshape to recover the 2-D structure.
N = round(np.sqrt(len(ys)))    # N = 21
x1_grid = x1.reshape(N, N)     # (21, 21)
x2_grid = x2.reshape(N, N)
frac_err_grid = frac_err.reshape(N, N)

plt.figure(figsize=(6, 5))
contourf = plt.tricontourf(x1, x2, frac_err, levels=50, vmin=0, vmax=5)
contours = plt.tricontour(x1, x2, frac_err, levels=50, colors='k', linewidths=0.5, vmin=0, vmax=5)
plt.clabel(contours, fmt='%.1f', fontsize=7)
plt.colorbar(contourf, label='|model - data| / |data|')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Task 1 - Absolute fractional error of linear regression fit')
plt.axis('scaled')
plt.tight_layout()
plt.savefig('task1_fractional_error.png', dpi=150)

# # Prints
# ###############################################################################
# param_labels = [
#     'c00', 'c10', 'c01',
#     'c20', 'c02', 'c11',
#     'c30', 'c03', 'c21', 'c12',
#     'c40', 'c04', 'c31', 'c13', 'c22',
# ]

# print("cvxpy status:", prob.status)
# print("Optimal c parameters:")
# print(f"{'Parameter':<10} {'Value cvxpy':>20} {'Value gradient descent':>22} {'|difference|':>15}")
# print("-" * 80)
# for label, val_opt, val_gd in zip(param_labels, out_cvxpy, out_grad):
#     print(f"{label:<10} {val_opt:>20.6f} {val_gd:>22.6f} {abs(val_opt - val_gd):>15.2e}")

# print(f"\nAs array cvxpy:             {out_cvxpy}")
# print(f"\nAs array gradient descent:  {out_grad}")   
# print(f"\nGradient descent converged in {len(thetas)} iterations.")
# print(f"Max parameter difference vs. cvxpy: "
#       f"{np.max(np.abs(out_cvxpy - out_grad)):.2e}")

# print(f"\nLeast-squares loss     : {loss(out_cvxpy , X, ys):.4f}")
# print(f"RMSE                     : {np.sqrt(2 * loss(out_cvxpy , X, ys) / len(ys)):.4f}")
# print(f"Mean |fractional error|  : {frac_err.mean() * 100:.2f} %")
# print(f"Median |fractional error|: {np.median(frac_err) * 100:.2f} %")








##########################################################################################################
# TASK 2. NEURONAL NETWORK 
##########################################################################################################

class Network:
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # Return the output of the network if "a" starts as the initial input vector

        # Hidden layers (sigmoid)
        for (b, w) in zip(self.biases[:-1], self.weights[:-1]):
             a = sigmoid(w @ a + b) # sigmoid function 
        # Output layer (linear)
        b, w = self.biases[-1], self.weights[-1]
        a = w @ a + b   # linear function
        return a # return a vector of activations from the output layer


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for (x, y) in mini_batch:
            # backprop method needs to calculate terms in the gradient calculation
            (grad_b_term, grad_w_term) = self.backprop(x, y)
            grad_b = [gb+gbt for (gb, gbt) in zip(grad_b, grad_b_term)]
            grad_w = [gw+gwt for (gw, gwt) in zip(grad_w, grad_w_term)]

        self.biases = [b - (eta/len(mini_batch)) * gb for (b, gb) in zip(self.biases, grad_b)]
        self.weights = [w - (eta/len(mini_batch))*gw for (w, gw) in zip(self.weights, grad_w)]

    def backprop(self, x, y):
        """Return a tuple ``(grad_b, grad_w)`` representing the
        gradient for the cost function C_x.  ``grad_b`` and
        ``grad_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x # input layer the activation vector is the input vector
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        # Hidden layers (sigmoid)
        for (b, w) in zip(self.biases[:-1], self.weights[:-1]):
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)      # sigmoid function (hidden layers)
            activations.append(activation)
        # Output layer (linear)
        b, w = self.biases[-1], self.weights[-1]
        z = w @ activation + b
        zs.append(z)
        activation = z                 
        activations.append(activation)


        # backward pass
        delta = self.loss_derivative(activations[-1], y) * 1 # (1 is the derivative of z)
        grad_b[-1] = delta 
        grad_w[-1] = delta@activations[-2].transpose()
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = (self.weights[-l+1].transpose() @ delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = delta @ activations[-l-1].transpose()
        return(grad_b, grad_w)

    def loss_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x /
        partial a for the output activations."""
        return output_activations - y 
    
    # Stochastic Gradient Descent Method
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    # Maximum ssum of squares function
    def compute_mse(self, data):
        return np.mean([0.5 * np.linalg.norm(self.feedforward(x) - y)**2 for (x, y) in data])
    
def sigmoid(z):
    # The sigmoid function
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # Derivative of the sigmoid function.
    return sigmoid(z)*(1-sigmoid(z))


# Reformat as list of (x, y) column vectors
data = [(x.reshape(2, 1), np.array([[y]])) for x, y in zip(xs, ys)]
np.random.seed(25221241)

eta = 0.15
epochs = 5000
batch_size = 3

# # Testing the different topologies
# topologies = [
#     [2, 3, 1],       # 13 params
#     [2, 4, 1],       # 17 params
#     [2, 5, 1],       # 21 params
#     [2, 2, 2, 1],    # 15 params
#     [2, 2, 3, 1],    # 19 params
#     [2, 3, 2, 1],    # 20 params
#     [2, 2, 2, 2, 1], # 21 params
# ]

# best_loss = np.inf
# best_net = None
# best_topology = None

# for sizes in topologies:
#     for trial in range(10):   # 10 random initialisations per topology
#         net = Network(sizes)
#         net.SGD(data, epochs=epochs, mini_batch_size=batch_size, eta=eta)
#         current_loss = net.compute_mse(data)
#         if current_loss < best_loss:
#             best_loss = current_loss
#             best_net = net
#             best_topology = sizes

# for (w, b) in zip(best_net.weights, best_net.biases):
#     print(w)
#     print(b)

# Best topology found: [2, 5, 1] (21 parameters)
best_topology = [2, 5, 1]
best_net = Network(best_topology)
best_net.SGD(data, epochs, batch_size, eta)

for (w, b) in zip(best_net.weights, best_net.biases):
    print(w)
    print(b)


y_pred_nn = np.array([best_net.feedforward(x.reshape(2,1))[0,0] for x in xs])
frac_err_nn = np.abs(y_pred_nn - ys) / np.abs(ys)

plt.figure(figsize=(6, 5))
contourf2 = plt.tricontourf(x1, x2, frac_err_nn, levels=50, vmin=0, vmax=5)
contours2 = plt.tricontour(x1, x2, frac_err_nn, levels=50, colors='k', linewidths=0.5, vmin=0, vmax=5)
plt.clabel(contours2, fmt='%.1f', fontsize=7)
plt.colorbar(contourf2, label='|model - data| / |data|')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Task 2 - Absolute fractional error of best neural network')
plt.axis('scaled')
plt.tight_layout()
plt.savefig('task2_fractional_error.png', dpi=150)


# # Prints
# ###############################################################################
# print(f"Best topology: {best_topology}")
# print(f"Total parameters: {sum(w.size + b.size for w, b in zip(best_net.weights, best_net.biases))}")
# print(f"MSE loss: {best_loss:.4f}")
# print(f"RMSE: {np.sqrt(2 * best_loss):.4f}")
# print(f"Mean |fractional error|: {frac_err_nn.mean() * 100:.2f} %")
# print(f"Median |fractional error|: {np.median(frac_err_nn) * 100:.2f} %")
