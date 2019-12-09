# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"

if not tf.test.is_gpu_available():
    print("No GPU was detected. CNNs can be very slow without a GPU.")

import os
import shutil
import numpy as np
import talos


class transfer_learning:
    '''
    Takes in base and top model and transfer learns them on the given dataset
    '''
    def __init__(self, name, n_classes, batch_size,
                 base_epochs, tune_epochs, image_path):
        self.__model_name = name
        self.__n_classes = n_classes
        self.model = None #Init model
        self.test_loss = None
        self.test_accuracy = None
        self.history = None
        self.__BATCH_SIZE = batch_size
        self.__base_epochs = base_epochs
        self.__tune_epochs = tune_epochs
        self.__image_path = image_path

    def __make_dirs(self, path, clean=False):
        if not os.path.exists(path):
            os.makedirs(path)
        else:  # Empty the directory
            if clean == True:
                shutil.rmtree(path)
        return

    def lr_schedule(self, epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        learning_rate = 0.2
        if epoch > 10:
            learning_rate = 0.02
        if epoch > 20:
            learning_rate = 0.01
        if epoch > 50:
            learning_rate = 0.005

        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate


    def _get_callbacks(self, train=True, name=""):

        callback_name= ""
        if train:
            callback_name = self.__model_name + "_train_" + name
        else:
            callback_name = self.__model_name

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(callback_name + "finetuned.h5", save_best_only=True)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        csv_logger_cb = tf.keras.callbacks.CSVLogger(callback_name + '_training.log')
        lr_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)

        tensorboard_dir = "logs/tensorboard"
        self.__make_dirs(tensorboard_dir)

        def get_run_logdir():
            import time
            run_id = time.strftime( callback_name + "_%Y_%m_%d-%H_%M_%S")
            return os.path.join(tensorboard_dir, run_id)

        run_logdir = get_run_logdir()
        tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

        callback_list = []
        if train:
            callback_list = [tensorboard_cb, checkpoint_cb, reduce_lr_cb, csv_logger_cb]
        else:
            callback_list = [tensorboard_cb, checkpoint_cb, early_stopping_cb, csv_logger_cb]
        return callback_list


    def __get_model__(self, base_model):
        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        reg = tf.keras.layers.BatchNormalization()(avg)
        class_output = tf.keras.layers.Dense(self.__n_classes, activation="softmax")(reg)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=[class_output])
        return model

    def __transfer_learn_generator__(self, base_model, model, train_gen, validation_gen):
        # Let the training settle
        print("Freezing all base layers")
        for layer in base_model.layers:
            layer.trainable = False

        adam = tf.keras.optimizers.Adam(lr=0.01)  # Higher lerarning rate
        model.compile(adam, loss=['categorical_crossentropy'], metrics=['accuracy'])

        history = model.fit_generator(train_gen, epochs=self.__base_epochs, workers=8,
                                      steps_per_epoch=train_gen.samples // self.__BATCH_SIZE,
                                      shuffle=True,
                                      validation_data=validation_gen,
                                      callbacks=self._get_callbacks(name="base"))

        # Unfreeze the layers from the base model
        print("Unfreezing all base layers")
        for layer in base_model.layers:
            layer.trainable = True

        adam = tf.keras.optimizers.Adam(lr=0.001)  # Lower the lerarning rate lerarning rate
        model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit_generator(train_gen, epochs=self.__tune_epochs, workers=8,
                                      steps_per_epoch=train_gen.samples // self.__BATCH_SIZE,
                                      shuffle=True,
                                      validation_data=validation_gen,
                                      callbacks=self._get_callbacks(name="finetune"))

        return model, history

    def train_generator(self, base_model, train_gen, validation_gen):
        self.model = self.__get_model__(base_model)
        self.model, self.history = self.__transfer_learn_generator__(base_model, self.model, train_gen, validation_gen)
        self.plot_training()
        return self.model, self.history

    def plot_training(self):

        import pandas as pd;
        import pickle
        import matplotlib.pyplot as plt

        pd.DataFrame(self.history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca()#.set_ylim(0, 1) # set the vertical range to [0-1]
        plt.title(self.__model_name  + " Transfer Learning")
        plt.show()
        savePath = os.path.join(self.__image_path, self.__model_name + "_history.png")
        plt.savefig(savePath)
        pickle.dump(self.history.history,
                    open(self.__model_name + "_history_" + "finetune.p", "wb"))

        return True

    def evaluate(self, test_gen, load_best=True):
        from sklearn.metrics import classification_report, confusion_matrix

        if load_best:
            best_model_filename = self.__model_name + "_train_finetunefinetuned.h5"
            self.model = tf.keras.models.load_model(best_model_filename)
            print("Loading {}".format(best_model_filename))

        Y_pred = self.model.predict_generator(test_gen, test_gen.samples//self.__BATCH_SIZE)
        print(Y_pred)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(test_gen.classes, y_pred))

        self.test_loss, self.test_accuracy = self.model.evaluate_generator(test_gen, test_gen.samples//self.__BATCH_SIZE,
                                                                           callbacks=self._get_callbacks(train=False))
        print("Test accuracy is: {}".format(self.test_accuracy))
        return self.test_loss, self.test_accuracy
