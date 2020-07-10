Original file is located at
    https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%202%20-%20Handwriting%20Recognition/Exercise2-Question.ipynb
"""

# Questions 4,5,6 JULY 8th, 2020

import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train  = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

classifications = model.predict(x_test)

import random
import numpy as np

random_number= random.randint(0,10000)

print('Classification probabilities of random number:')
print(classifications[random_number])
print('The random number is:')
print(random_number)
print('The best classification is:')
np.argmax(classifications[random_number])

import matplotlib.pyplot as plt
plt.imshow(x_test[random_number])

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

x=['0','1','2','3','4','5','6','7','8','9']
y = classifications[random_number]

plt.bar(x,y)
plt.xlabel('Categories')
plt.ylabel("Values")
plt.title('Categories Bar Plot')
plt.show()

