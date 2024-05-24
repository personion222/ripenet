from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, DetCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import load_dat
import keras

dat = load_dat.load_dat("data_fin/ann", 1)[0]

ripe_model = keras.saving.load_model("single_nets/model2.keras", compile=True)
ripe_model.summary()

true = list()
pred = list()
pred_round = list()

for ann in dat:
    true.append(ann.ripeness)
    frame = ann.img.reshape((1, 64, 64, 3))
    pred.append(ripe_model.predict(frame, verbose=None)[0, 0])
    pred_round.append(round(pred[-1]))

true = np.asarray(true)
pred = np.asarray(pred)
pred_round = np.asarray(pred_round)

RocCurveDisplay.from_predictions(true, pred)
plt.show()

PrecisionRecallDisplay.from_predictions(true, pred)
plt.show()

ConfusionMatrixDisplay.from_predictions(true, pred_round)
plt.show()

DetCurveDisplay.from_predictions(true, pred)
plt.show()
