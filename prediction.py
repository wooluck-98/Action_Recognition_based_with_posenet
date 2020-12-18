import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers.wrappers import TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models  
import numpy as np
import json
import cv2

LABELS = [
 'back_cross_step',
 'clap',
 'cross_heel',
 'downup',
 'knee_up',
 'lunge',
 'openclose',
 'shoulder',
 'side',
 'stand_clap',
 'sting',
 'wave'
]
WINDOW  = 10
n_classes = len(LABELS)

pose = {}
for i, label in enumerate(LABELS):
    pose[i] = label


#prediction
action_model = models.load_model("best_model/model-morning-stretch.h5")
action_model.load_weights("best_weight/val-morning-stretch.hdf5")

def predictLSTM(model, lst, none):
    pred = model.predict(np.array([lst]))

    label = np.argmax(np.mean(pred[0], axis=0))
    if np.mean(pred[0], axis=0)[label] < 0.5:
        return lst[1:], none
    return lst[1:], label

frame_stack = []
cap = cv2.VideoCapture(0)
cap.set(3, 512)
cap.set(4, 512)

while True:
    ret, frame = cap.read()
    cv2.imshow('action_recognition', frame)

    frame_stack.append(frame)
    if len(morning_list)==10:
        frame_stack, label = predictLSTM(action_model, frame_stack, 12)
     
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()