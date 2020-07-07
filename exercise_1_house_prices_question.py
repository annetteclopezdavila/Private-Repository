# -*- coding: utf-8 -*-
"""Exercise_1_House_Prices_Question.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/annetteclopezdavila/85cfc72d2f6c481b523f6b4460d6ed0a/exercise_1_house_prices_question.ipynb
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.
"""

#MY CODE FOR THE EXERCISE

#import
import tensorflow as tf
import numpy as np
from tensorflow import keras
#create one layer neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#optimizer, loss
model.compile(optimizer='sgd', loss='mean_squared_error')
#data
x = np.array([1.0,2.0,3.0,4.0,5.0,6.0], dtype=float)
y = np.array([1.0,1.5,2.0,2.5,3.0,3.5], dtype=float)
#fit model
model.fit(x, y, epochs=1000)
#if X=7 what is Y?
print(model.predict([7.0]))

