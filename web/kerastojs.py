import os
import tensorflowjs as tfjs
import tensorflow as tf
import keras

model = keras.saving.load_model("temp.h5", compile=True)
model.summary()
tf.saved_model.save(model, "pymodels/ripe_single")
# tfjs.converters.save_keras_model(model, "models/ripe_single")
