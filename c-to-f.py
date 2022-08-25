import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# training data for simple temperature conversion example (F --> K)
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

# build the model 
l0 = tf.keras.layers.Dense( # dense layers are connected to every neuron in the previous layer
    units=1, # how many neurons (or internal tunable variables) in this layer
    input_shape=[1] # shape of the input, in this case a 1-dimensional array with one member
    )
model = tf.keras.Sequential([l0])

# compile the model 
model.compile(
    loss='mean_squared_error',  #how should it calculate the loss (aka the cost)?
    optimizer=tf.keras.optimizers.Adam(0.1) # what method should be used to tune the neurons given a loss?
)

# train the model. 
# model.fit() returns a history object detailing the progress of training
history = model.fit(
    celsius_q, # inputs (aka features) of the training data
    fahrenheit_a, # outputs (aka labels) of the training data
    epochs=500, # number of times the calculate-compare-adjust cycle will run
    verbose=False # how noisy will the output be during training? 0 - silent; 1 - progress bar; 2 - current epoch
)
print('Model training is complete')

# display training statistics
plt.plot(history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()

# predict a few values and compare
c_tests = [-273, -100, 25, 500]
f_real = [-459, -148, 77, 932]
for i in range(len(c_tests)): 
    print(f'The model predicts {model.predict([c_tests[i]])}. The correct value is {f_real[i]}')