import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    dice = tf.reduce_mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def dice_loss(y_true, y_pred, smooth=1):
    dice = dice_coef(y_true, y_pred, smooth)
    return 1 - dice

def Checkpoints(name:str,path:str="C:/Om/Checkpoints/"):
    filepath = f"../Om/Checkpoints/{name}/{name}.cpkt"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_freq='epoch')
    print(f"ModelCheckpoint({filepath}, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_freq='epoch')")
    return checkpoint

