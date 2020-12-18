import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers.wrappers import TimeDistributed
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np

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

XchTrain = np.load("trainX.npy")
YTrain = np.load("trainY.npy")

# making test and train labels one hot
YintTrain = np.int64(YTrain)
YhotTrain = np.zeros((YTrain.shape[0], n_classes))
YhotTrain[np.arange(YTrain.shape[0]), YintTrain] = 1
YhotTrain = np.repeat(YhotTrain[:, :, np.newaxis], WINDOW, axis=2)
YhotTrain = np.swapaxes(YhotTrain, 1, 2)

# Validation data
XchVal = np.load("valX.npy")
YVal = np.load("valY.npy")

YintVal = np.int64(YVal)
YhotVal = np.zeros((YVal.shape[0], n_classes))
YhotVal[np.arange(YVal.shape[0]), YintVal] = 1
YhotVal = np.repeat(YhotVal[:, :, np.newaxis], WINDOW, axis=2)
YhotVal = np.swapaxes(YhotVal, 1, 2)

#lstm_model_define
def get_model():
    model = Sequential([
        TimeDistributed(Conv1D(16,3, activation='relu', padding = "same"),input_shape=XchTrain.shape[1:]),
        TimeDistributed(BatchNormalization()),
        BatchNormalization(),
        TimeDistributed(Flatten()),
        LSTM(32, return_sequences=True,  unit_forget_bias=1.0,dropout=0.2),
        TimeDistributed(Dense(n_classes,activation='softmax'))   
    ])

    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
    return model

#train
model = get_model()
filepath= "weights/" +"lstm_model" + "-{epoch:02d}-{val_accuracy:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model_history = model.fit(XchTrain, YhotTrain, epochs=100, batch_size=2, callbacks=callbacks_list, validation_data = (XchVal, YhotVal))
model.save("model/lstm_model.h5")