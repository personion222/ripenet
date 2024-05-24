import matplotlib.pyplot as plt
import numpy as np
import load_dat
import keras

dat = load_dat.load_dat("data_fin/ann", 1)[0]

ripe_model = keras.saving.load_model("single_nets/model2.keras", compile=True)
ripe_model.summary()

true = list()
pred = list()

for ann in dat:
    true.append(ann.ripeness)
    frame = ann.img.reshape((1, 64, 64, 3))
    pred.append(ripe_model.predict(frame, verbose=None)[0, 0])

ripe_vals = list()
unripe_vals = list()

for idx, t in enumerate(true):
    p = pred[idx]
    if t:
        ripe_vals.append(p)
    else:
        unripe_vals.append(p)

print(ripe_vals)

plt.hist(ripe_vals, 10, range=(0, 1))
plt.show()

plt.hist(unripe_vals, 10, range=(0, 1))
plt.show()
