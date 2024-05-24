from load_dat import load_dat
from tensorflow import keras
import matplotlib as mpl
import tensorflow as tf
import pandas as pd
import numpy as np

TRAIN_TEST = 0.8
BATCH_SIZE = 1
EPOCHS = 128

mpl.use("WebAgg")

np.set_printoptions(precision=2, edgeitems=8, linewidth=200)

ann_dat = load_dat("data_fin/ann", 1)[0]
dat_count = len(ann_dat)

model = keras.models.Sequential([
    keras.layers.Input((64, 64, 3)),
    keras.layers.Conv2D(8, (7, 7), activation="relu", use_bias=True),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(16, (5, 5), activation="relu", use_bias=True),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, (3, 3), activation="relu", use_bias=True),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, (1, 1), activation="relu", use_bias=True),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l1(), use_bias=True),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l1(), use_bias=True),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.l1(), use_bias=True)
])

model.build((None, 64, 64, 3))
model.summary()

model_in = np.empty((dat_count,) + ann_dat[0].dims, dtype=np.half)
model_out = np.empty(dat_count, dtype=np.half)

for idx, ann in enumerate(ann_dat):
    model_in[idx, :] = ann.img
    model_out[idx] = ann.ripeness

# print(model_in[:, 1])
# print(model_out[:, 1])

model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(0.00001), metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.MeanAbsoluteError(), keras.metrics.AUC()])

ckpt_path = "ckpt.weights.h5"

save_best = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_binary_accuracy", save_best_only=True, mode="max", save_weights_only=True)

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

model.save("single_nets/model3.keras")

hist_df = pd.DataFrame(history.history)

csv_writer = open("single_nets/history3.csv", mode='w')
hist_df.to_csv(csv_writer)
csv_writer.close()
