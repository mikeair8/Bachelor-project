import sys
import numpy as np
import pandas as pd
import keras.models
import tensorflow as tf
from tensorflow.keras import layers,models,callbacks,optimizers
import matplotlib
import matplotlib.pyplot as plt
import gc
class Objective:
    def __init__(self,
                 x_train=None,
                 y_train=None,
                 x_valid=None,
                 y_valid=None,
                 patience=50,
                 batch_size=1,
                 epochs=100,
                 loss=None,
                 dropout_prob=0.2):
        self.x_train=x_train,
        self.y_train=y_train,
        self.x_valid = x_valid,
        self.y_valid = y_valid,
        self.patience=patience
        self.batch_size=batch_size
        self.epochs=epochs
        self.dropout_prob=dropout_prob
        self.loss=loss
        self.maxseed=0
    def Evaluate(self, pos, queue, i):
        model = self.create_model(pos)
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.75, patience=30)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        history = model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid),
                            batch_size=self.batch_size, epochs=self.epochs, callbacks=[lr_scheduler,early_stopping_cb], verbose=0)
        val_losses = np.array(history.history["val_loss"])
        train_losses = np.array(history.history["loss"])
        index_min = np.argmin(val_losses)
        val=val_losses[index_min]
        train=train_losses[index_min]
        queue.put([val, train, i,model,history])
        print("Particle",pos,"validation MAE", val)
        del model, history,val,train, lr_scheduler, early_stopping_cb

    def create_model(self, pos):
        activation_met="softplus"
        ann = models.Sequential()
        ann.add(layers.Dense(pos[0],kernel_initializer="he_uniform", activation=activation_met, input_shape=(30,)))
        ann.add(layers.BatchNormalization())
        for i in range(1,len(pos)):
            ann.add(layers.Dense(pos[i], activation=activation_met,kernel_initializer="he_uniform"))
            ann.add(layers.BatchNormalization())
        ann.add(layers.Dense(1, activation="relu",kernel_initializer="he_uniform"))
        ann.compile(optimizer=optimizers.Adam(0.01), loss=self.loss)
        return ann



##The goal of the Objective.Evaluate() Function is to measure the quality of each particle position