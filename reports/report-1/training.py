@tf.function
def train_step(self):
    trainable_params = self.model.trainable_variables
    with tf.GradientTape() as model_tape:
        resloss = 0
        for constraint in self.constraints:
            domain = constraint.domain
            residual = constraint.residual
            stack = self.grads(domain)
            resloss += self.loss(
                residual(domain, stack),
                tf.constant(0.0, dtype=tf.float32))
    gradients = model_tape.gradient(resloss, trainable_params)
    self.optimizer.apply_gradients(zip(gradients, trainable_params))
    return resloss