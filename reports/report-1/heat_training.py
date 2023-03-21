heat1 = tf.keras.Sequential([
    tf.keras.layers.Input((2,)),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(1)
])

epochs = 25000
learning_rate = 5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.MeanSquaredError()

pinn = PINN(heat1)
pinn.constraints = [pde_con, bc0_con, bc1_con, ic_con]
pinn.compile(optimizer, loss, ['mae'], order=2)
pinn.fit(epochs=epochs)