import tensorflow as tf

class Constraint:
    def __init__(self, domain, residual):
        self.domain = tf.constant(domain, dtype=tf.float32)
        self.residual = residual

class PINN():
    
    def __init__(self, model):
        self.model = model
        self._compiled = False
        self._constraints = None
        
    @property
    def constraints(self):
        return self._constraints
    
    @constraints.setter
    def constraints(self, constraints):
        self._constraints = constraints
    
    def save(self, model_name = 'pinn_model.h5'):
        self.model.save(model_name)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)
    
    def compile(self, optimizer, loss, metrics, order = 1):
        self.optimizer = optimizer
        self.loss = loss
        self.order = order
        self.predict_U_hat = grads(self.model, self.order)
        
        self._compiled = True
    
    @tf.function
    def __call__(self, phi, training=False):
        return self.model(phi, training=training)
    
    
    @tf.function
    def train_step(self):
        trainable_params = self.model.trainable_variables
        with tf.GradientTape() as model_tape:
            resloss = 0
            for constraint in self.constraints:
                domain, residual = constraint.domain, constraint.residual
                U_hat = self.predict_U_hat(domain)
                resloss += self.loss(residual(domain, U_hat), tf.constant(0.0, dtype=tf.float32))
        
        gradients = model_tape.gradient(resloss, trainable_params)
        self.optimizer.apply_gradients(zip(gradients, trainable_params))
        return resloss
            
    
    def fit(self, constraints=None, epochs=1):
        if constraints is None: constraints = self.constraints 
        assert self._compiled, 'PINN object must be compiled before training.'
        for epoch in range(epochs):
            
            loss_value = self.train_step()
            
            if epoch % 1000 == 0:
                print(
                    "Training loss at epoch %d: %.4f"
                    % (epoch, float(loss_value))
                )

def grads(y, order):
    
    def _grads(x):
        size = x.shape[0]
        dim_x = x.shape[1]
        stack = []
        def _grad(y, x, order):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                if order == 1: stack.append(y(x))
                if order > 1: _grad(y, x, order - 1)
                components = tf.unstack(
                    tf.reshape(stack[-1], (size, -1)), 
                    axis=1
                )
                
            derivatives = tf.stack(
                [tape.gradient(component, x) for component in components],
                axis = 2
            )
            stack.append(
               tf.reshape(derivatives, (size,) + order * (dim_x,) + (-1,)) 
            )
            del tape
        _grad(y, x, order)
        return stack
    
    return _grads