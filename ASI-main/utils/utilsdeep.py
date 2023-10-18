
import tensorflow as tf
from tensorflow import keras
from numpy.random import seed
from tensorflow.keras.callbacks import Callback

seed(23)
tf.compat.v1.set_random_seed(23)


class CustomLearningRateScheduler(Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""

    lr = 50 ** (-0.5) * tf.math.minimum((epoch+1) ** (-0.5), (epoch+1) * (100 ** (-1.5)))

    return lr


# Callback for loss logging per epoch

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


history = LossHistory()

# Callback for early stopping the training
early_stopping_2_output = keras.callbacks.EarlyStopping(monitor='val_main_output_loss',
                                                        min_delta=0,
                                                        patience=10,
                                                        verbose=0, mode='auto')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=20,
                                               verbose=0, mode='auto')

early_stopping_acc = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                   min_delta=0,
                                                   patience=10,
                                                   verbose=0, mode='auto')
