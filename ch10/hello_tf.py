#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#TensorFlow example based on https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf
import matplotlib.pyplot as plt 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10)
	])

# specify the loss function (cross entropy)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# iniitialize model to start training
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# train the model and then run with test data
model.fit(x_train, y_train, epochs=3)
model.evaluate(x_test,  y_test, verbose=2)

# visualize few example results
fig,axs = plt.subplots(1,4)
for i in range(4):
    axs[i].imshow(x_test[120+i],cmap='gray_r'); 
    axs[i].set_xlabel(y_test[120+i],size=22,weight='bold');
    axs[i].set_xticks([]); axs[i].set_yticks([])
plt.tight_layout();
plt.show(block=True);

print("Done")
