"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
#Import the libraries
import numpy as np

class Optimizer:
    """
    Base class for optimizers
    
    Args:
        learning_rate (float): Learning rate    

    Attributes:
        lr (float): Learning rate
    """
    def __init__(self, learning_rate):
        """
        Initialize the optimizer
        
        Args:
            learning_rate (float): Learning rate
        """
        self.lr = learning_rate

    def update(self, layers, grads):
        """
        Update the weights
        
        Args:
            layers (list): List of layers
            grads (list): List of gradients
        """
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    
    Args:
        learning_rate (float): Learning rate
    """
    def update(self, layers, grads):
        for i, layer in enumerate(layers):
            dW, db = grads[i]
            layer.W -= self.lr * dW
            layer.b -= self.lr * db

class Momentum(Optimizer):
    """
    Momentum Gradient Descent Optimizer
    
    Args:
        learning_rate (float): Learning rate
        
        
    Attributes:
        v_W (dict): Dictionary of velocity terms for weights
        v_b (dict): Dictionary of velocity terms for biases
    """
    def __init__(self, learning_rate, gamma=0.9):
        """
        Initialize the optimizer
        
        Args:
            learning_rate (float): Learning rate
            gamma (float): Momentum parameter
        """
        super().__init__(learning_rate)
        self.gamma = gamma
        self.v_W = {}
        self.v_b = {}

    def update(self, layers, grads):
        """
        Update the weights
        
        Args:
            layers (list): List of layers
            grads (list): List of gradients
        """
        for i, layer in enumerate(layers):
            dW, db = grads[i]
            # Initialize velocity dictionaries if running for the first time
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)
            # Update accumulated velocities
            self.v_W[i] = self.gamma * self.v_W[i] + self.lr * dW
            self.v_b[i] = self.gamma * self.v_b[i] + self.lr * db
            # Update weights
            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]

class NAG(Optimizer):
    """
    Nesterov Accelerated Gradient
    
    Args:
        learning_rate (float): Learning rate
        
    Attributes:
        v_W (dict): Dictionary of velocity terms for weights
        v_b (dict): Dictionary of velocity terms for biases
    """
    def __init__(self, learning_rate, gamma=0.9):
        """
        Initialize the optimizer
        
        Args:
            learning_rate (float): Learning rate
            gamma (float): Momentum parameter
        """
        super().__init__(learning_rate)
        self.gamma = gamma
        self.v_W = {}
        self.v_b = {}

    def update(self, layers, grads):
        """
        Update the weights
        
        Args:
            layers (list): List of layers
            grads (list): List of gradients
        
        Returns:
            None
        """
        for i, layer in enumerate(layers):
            dW, db = grads[i]
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)
            # Update velocity
            self.v_W[i] = self.gamma * self.v_W[i] + self.lr * dW
            self.v_b[i] = self.gamma * self.v_b[i] + self.lr * db
            # Nesterov update
            layer.W -= (self.gamma * self.v_W[i] + self.lr * dW)
            layer.b -= (self.gamma * self.v_b[i] + self.lr * db)

class RMSprop(Optimizer):
    """
    RMSprop Optimizer
    
    Attributes:
        s_W (dict): Dictionary of squared gradients for weights
        s_b (dict): Dictionary of squared gradients for biases
        
    Args:
        learning_rate (float): Learning rate
    """
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        """
        Initialize the optimizer
        
        Args:
            learning_rate (float): Learning rate
            beta (float): Decay rate
            epsilon (float): Small number
        
        Returns:
            None
        """
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.s_W = {}
        self.s_b = {}

    def update(self, layers, grads):
        """
        
        Args:
            layers (list): List of layers
            grads (list): List of gradients
            
        Returns:
            None
        """
        for i, layer in enumerate(layers):
            dW, db = grads[i]
            if i not in self.s_W:
                self.s_W[i] = np.zeros_like(layer.W)
                self.s_b[i] = np.zeros_like(layer.b)
            # Moving average of squared gradients
            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * (dW ** 2)
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (db ** 2)
            # Update weights
            layer.W -= (self.lr / (np.sqrt(self.s_W[i]) + self.epsilon)) * dW
            layer.b -= (self.lr / (np.sqrt(self.s_b[i]) + self.epsilon)) * db

class Adam(Optimizer):
    """
    Adam Optimizer
    
    Attributes:
        m_W (dict): Dictionary of first moment terms for weights
        m_b (dict): Dictionary of first moment terms for biases
        v_W (dict): Dictionary of second moment terms for weights
        v_b (dict): Dictionary of second moment terms for biases
        
    Args:
        learning_rate (float): Learning rate
        
    Returns:
        None
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the optimizer
        
        Args:
            learning_rate (float): Learning rate
            beta1 (float): Decay rate for first moment
            beta2 (float): Decay rate for second moment
            epsilon (float): Small number
            
        Returns:
            None
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W, self.m_b = {}, {} # First moment
        self.v_W, self.v_b = {}, {} # Second moment
        self.t = 0  #Time

    def update(self, layers, grads):
        """
        Update the weights
        
        Args:
            layers (list): List of layers
            grads (list): List of gradients
            
        Returns:
            None
        """
        self.t += 1
        for i, layer in enumerate(layers):
            dW, db = grads[i]
            if i not in self.m_W:
                self.m_W[i] = np.zeros_like(layer.W)
                self.m_b[i] = np.zeros_like(layer.b)
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)
            # Update first derivative
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            # Update second derivative
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dW ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db ** 2)
            # Bias correction
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            # Update weights
            layer.W -= (self.lr / (np.sqrt(v_W_hat) + self.epsilon)) * m_W_hat
            layer.b -= (self.lr / (np.sqrt(v_b_hat) + self.epsilon)) * m_b_hat

class Nadam(Optimizer):
    """
    Nadam Optimizer
    
    Attributes:
        m_W (dict): Dictionary of first moment terms for weights
        m_b (dict): Dictionary of first moment terms for biases
        v_W (dict): Dictionary of second moment terms for weights
        v_b (dict): Dictionary of second moment terms for biases
    
    Args:
        learning_rate (float): Learning rate
        
    Returns:
        None
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the optimizer
        
        Args:
            learning_rate (float): Learning rate
            beta1 (float): Decay rate for first moment
            beta2 (float): Decay rate for second moment
            epsilon (float): Small number
            
        Returns:
            None
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W, self.m_b = {}, {}
        self.v_W, self.v_b = {}, {}
        self.t = 0

    def update(self, layers, grads):
        """
        Update the weights
        
        Args:
            layers (list): List of layers
            grads (list): List of gradients
            
        Returns:
            None
        """
        self.t += 1
        for i, layer in enumerate(layers):
            dW, db = grads[i]
            if i not in self.m_W:
                self.m_W[i], self.m_b[i] = np.zeros_like(layer.W), np.zeros_like(layer.b)
                self.v_W[i], self.v_b[i] = np.zeros_like(layer.W), np.zeros_like(layer.b)
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dW ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db ** 2)
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            # Nesterov update for momentum
            m_W_nesterov = self.beta1 * m_W_hat + ((1 - self.beta1) * dW) / (1 - self.beta1 ** self.t)
            m_b_nesterov = self.beta1 * m_b_hat + ((1 - self.beta1) * db) / (1 - self.beta1 ** self.t)
            layer.W -= (self.lr / (np.sqrt(v_W_hat) + self.epsilon)) * m_W_nesterov
            layer.b -= (self.lr / (np.sqrt(v_b_hat) + self.epsilon)) * m_b_nesterov

def get_optimizer(optimizer_name, learning_rate):
    """
    Get the optimizer class
    
    Args:
        optimizer_name (str): Name of the optimizer
        learning_rate (float): Learning rate
        
    Returns:
        Optimizer: Optimizer class
    """
    opt_map = {
        'sgd': SGD,
        'momentum': Momentum,
        'nag': NAG,
        'rmsprop': RMSprop,
        'adam': Adam,
        'nadam': Nadam
    }
    opt_class = opt_map.get(optimizer_name.lower(), SGD)
    return opt_class(learning_rate)