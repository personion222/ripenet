import matplotlib.pyplot as plt
from load_dat import load_dat
from tensorflow import keras
import matplotlib as mpl
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2


def cust_mse(y_true, y_pred):
    return tf.math.multiply(tf.math.square(tf.math.multiply(tf.math.subtract(y_true, y_pred), 64)), 0.02)


def pixel_loss(y_true, y_pred):
    return tf.math.multiply(tf.math.abs(tf.math.subtract(y_true, y_pred)), 64)


TRAIN_TEST = 0.8
BATCH_SIZE = 16
EPOCHS = 256

mpl.use("WebAgg")

np.set_printoptions(precision=2, edgeitems=8, linewidth=200)

ann_dat = load_dat("fab_qoi/ann", 0.01)[0]
dat_count = len(ann_dat)

# print(ann_dat[1].cropped)

'''
mod_img = np.random.choice(ann_dat)

print((int(mod_img.cont_box[0] * mod_img.dims[1]),
       int(mod_img.cont_box[1] * mod_img.dims[0])),
      (int((mod_img.cont_box[2] + mod_img.cont_box[0]) * mod_img.dims[1]),
       int((mod_img.cont_box[3] + mod_img.cont_box[1]) * mod_img.dims[0])))

print(mod_img.filename)

mod_img.img = cv2.rectangle(mod_img.img,
                            (int(mod_img.cont_box[0] * mod_img.dims[1]),
                             int(mod_img.cont_box[1] * mod_img.dims[0])),
                            (int((mod_img.cont_box[2] + mod_img.cont_box[0]) * mod_img.dims[1]),
                             int((mod_img.cont_box[3] + mod_img.cont_box[1]) * mod_img.dims[0])),
                            (30 / 255, 140 / 255, 255 / 255), 1)

plt.imshow(mod_img.img, interpolation="nearest")
plt.show()
'''

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation=keras.activations.relu, use_bias=True, input_shape=(64, 64, 3)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(24, (3, 3), activation=keras.activations.relu, use_bias=True),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu, use_bias=True),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(48, (3, 3), activation=keras.activations.relu, use_bias=True),
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu, use_bias=True),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation=keras.activations.selu, kernel_regularizer=keras.regularizers.l1(), use_bias=True),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=keras.activations.selu, kernel_regularizer=keras.regularizers.l1(), use_bias=True),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation=keras.activations.linear, kernel_regularizer=keras.regularizers.l1(), use_bias=True)
])

model.build((None, 64, 64, 3))
model.summary()
# print(tf.config.list_physical_devices('GPU'))

model_in = np.empty((dat_count,) + ann_dat[0].dims, dtype=np.half)
model_out = np.empty((dat_count, 4), dtype=np.half)

for idx, ann in enumerate(ann_dat):
    model_in[idx, :] = ann.img
    model_out[idx, :] = ann.cont_box

# print(model_in[:, 1])
# print(model_out[:, 1])

model.compile(loss=cust_mse, optimizer=keras.optimizers.Adam(0.001), metrics=[pixel_loss])

ckpt_path = "ckpt.weights.h5"

save_best = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True)

history = model.fit(
    x=model_in,
    y=model_out,
    batch_size=BATCH_SIZE,
    shuffle=True,
    epochs=EPOCHS,
    verbose=1,
    validation_split=TRAIN_TEST,
    callbacks=[save_best]
)

model.load_weights(ckpt_path)

model.save("model1n.keras")

hist_df = pd.DataFrame(history.history)

csv_writer = open("history1n.csv", mode='w')
hist_df.to_csv(csv_writer)
csv_writer.close()
