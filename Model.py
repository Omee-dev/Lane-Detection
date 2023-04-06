import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation,Input,concatenate,Dropout,Conv2DTranspose,UpSampling2D,BatchNormalization,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

def unet():
    input_size = (256,256,3)
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model( inputs,conv10)
    #model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def simple_model(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    outputs = Conv2D(2, (1, 1), activation='softmax', padding='same')(conv4)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def Unet1():
    # Define the input shape
    input_shape = (256, 256, 3)

    # Define the number of filters for each convolutional layer
    filters = [64, 128, 256, 512, 1024]

    # Define the input tensor
    inputs = Input(input_shape)

    # Encoder Path
    conv1 = Conv2D(filters[0], (3, 3), activation='relu', padding='same',name="conv1_1")(inputs)
    conv1 = Conv2D(filters[0], (3, 3), activation='relu', padding='same',name="conv1_1")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(filters[1], (3, 3), activation='relu', padding='same',name="conv2_1")(pool1)
    conv2 = Conv2D(filters[1], (3, 3), activation='relu', padding='same',name="conv2_2")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(filters[2], (3, 3), activation='relu', padding='same',name="conv3_1")(pool2)
    conv3 = Conv2D(filters[2], (3, 3), activation='relu', padding='same',name="conv3_2")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(filters[3], (3, 3), activation='relu', padding='same',name="conv4_1")(pool3)
    conv4 = Conv2D(filters[3], (3, 3), activation='relu', padding='same',name="conv4_2")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Bottom
    conv5 = Conv2D(filters[4], (3, 3), activation='relu', padding='same',name="conv5_1")(pool4)
    conv5 = Conv2D(filters[4], (3, 3), activation='relu', padding='same',name="conv5_2")(conv5)

    # Decoder Path
    up6 = Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), padding='same',name="convt_1")(conv5)
    concat6 = concatenate([up6, conv4])
    conv6 = Conv2D(filters[3], (3, 3), activation='relu', padding='same',name="convup1_1")(concat6)
    conv6 = Conv2D(filters[3], (3, 3), activation='relu', padding='same',name="convup1_2")(conv6)

    up7 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same',name="convt_2")(conv6)
    concat7 = concatenate([up7, conv3])
    conv7 = Conv2D(filters[2], (3, 3), activation='relu', padding='same',name="convup2_1")(concat7)
    conv7 = Conv2D(filters[2], (3, 3), activation='relu', padding='same',name="convup2_2")(conv7)

    up8 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same',name="convt_3")(conv7)
    concat8 = concatenate([up8, conv2])
    conv8 = Conv2D(filters[1], (3, 3), activation='relu', padding='same',name="convup3_1")(concat8)
    conv8 = Conv2D(filters[1], (3, 3), activation='relu', padding='same',name="convup3_2")(conv8)

    up9 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same',name="convt_4")(conv8)
    concat9 = concatenate([up9, conv1])
    conv9 = Conv2D(filters[0], (3, 3), activation='relu', padding='same',name="convup4_1")(concat9)
    conv9 = Conv2D(filters[0], (3, 3), activation='relu', padding='same',name="convup4_2")(conv9)


    outputs = Conv2D(2, (1, 1), activation='sigmoid')(conv9) #yesorno 1 else 2 for heatmap left r8 lane

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    model.summary()

    return model

def yolo_v2_model(input_shape=(256,256,3)):
    input = Input(input_shape)

    def blocks(x,filters:int,block_no:int,max:int=1):
        y= Conv2D ( filters,(3,3),strides = (1,1),padding='same',name=f'conv{block_no}_1')(x)
        y= BatchNormalization(name=f"bnorm{block_no}")(y)
        y=Activation('relu')(y)
        if max:
            y= MaxPooling2D(pool_size=(2,2),name=f"maxpool{block_no}")(y)
        return y

    x = blocks(input,32,1) #block 1
    x = blocks(x,64,2)      #block 2
    x = blocks(x,128,3)     #block 3
    x = blocks(x,256,4)     #block 4
    x = blocks(x,512,5)     #block 5
    x = blocks(x,1024,6,max=0)    #block 6

    x = Conv2D(8,(3,3),strides=(1,1),padding='same',name="output_conv")()
    output = Flatten(name="output")(x)

    model = tf.keras.Model(inputs=[input], outputs=[output])
    model.summary()
    return model



    