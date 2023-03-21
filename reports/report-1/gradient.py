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