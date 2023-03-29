import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


def load_data(path:str="D:/Data",sample:int = -1):

    images = []
    labels = []
    data_set=[] 
    df = path
    results=[]
    with open(df+"/list/train_gt.txt","r") as f:
        for line in f :
            #line = line.strip()
            data = line.split(" ")
            set1 = data[0].split("/")[1:]
            if(set1[0] == "driver_182_30frame"):
                results.append(data)
    results.sort()
    for data in results :
        exist=[int(x) for x in data[2:]]
        label_sep = data[1].split("/")
        img_path= df+data[0]
        label_path = df+data[1]
        dict_culane={
            "img_path":img_path ,
            "label_path":label_path ,
            "exist":exist 
        }
        data_set.append(dict_culane)
    if sample == -1:
        sample = len(data_set)    
    for img in data_set[:sample]:
        print_progress_bar(data_set.index(img),len(data_set[:sample]),"load_data")
        image = cv2.imread(img["img_path"])
        label = cv2.imread(img["label_path"], cv2.IMREAD_GRAYSCALE)

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels),data_set[:sample]

def preprocess_data(images, labels, target_size=(256, 256)):
    images = np.array([cv2.resize(img, target_size) for img in images])
    images = images.astype('float32') / 255.0
    labels = np.array([cv2.resize(label, target_size) for label in labels])
    labels = np.where(labels > 0, 1, 0)
    labels = np.expand_dims(labels, axis=-1)
    return images, labels


def preprocess_images(images, target_size=(256, 256)):
    #preprocessed_images = np.zeros((len(images), target_size[0], target_size[1], 3))
    preprocess_images = []
    for i in range(len(images)):
        print_progress_bar(i,len(images),"preprocess_images")
        img = cv2.resize(images[i], target_size)
        img = img.astype('float32') / 255.0
        preprocess_images.append(img)
    
    return np.array(preprocess_images)

def create_binary_masks(labels):
    binary_masks = []
    for i,label in enumerate(labels):
        print_progress_bar(i,len(labels),"creating_binary_masks")
        binary_mask = np.zeros(label.shape[:2], dtype=np.float32)
        coordinates = np.where(label != [0, 0, 0])
        binary_mask[coordinates[0], coordinates[1]] = 1
        binary_masks.append(binary_mask)
    binary_masks = np.array(binary_masks)
    binary_masks = binary_masks.reshape(binary_masks.shape[0], binary_masks.shape[1], binary_masks.shape[2], 1)
    
    return binary_masks

def split_data(images, labels, test_size=0.2, validation_size=0.2, random_state=42):
    # Split into training and testing sets
    print("spliting data")
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    
    # Split the training set into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=random_state)
    
    return x_train, x_val, x_test, y_train, y_val, y_test

def augment_data(x_train, y_train,batch_size=32):
    print("augmenting data")
    datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,horizontal_flip=True)
    datagen.fit(x_train)
    generator = datagen.flow(x_train, y_train, batch_size=batch_size)   
    return generator

def augment_data1(x_train, y_train, batch_size=32):
    # Create an ImageDataGenerator object with data augmentation parameters
    train_datagen = ImageDataGenerator(
        rotation_range=10, # Rotate the image randomly between 0 and 10 degrees
        width_shift_range=0.1, # Shift the image horizontally by up to 10% of the image width
        height_shift_range=0.1, # Shift the image vertically by up to 10% of the image height
        zoom_range=0.1, # Zoom the image in or out by up to 10%
        horizontal_flip=True, # Flip the image horizontally
        fill_mode='nearest' # Fill in any empty pixels with the nearest value
    )

    # Load the training set and apply data augmentation
    train_generator = train_datagen.flow(
        x_train, # Input images
        y_train, # Binary masks
        batch_size=batch_size,
        shuffle=True # Shuffle the data
    )

    return train_generator

def print_progress_bar(iteration, total,txt=""):
    length = 40
    iteration += 1
    percent = (100 * iteration) // (total * 99/100)
    filled_length = int(length * percent / 100)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print('\r%s |%s %s/%s %s' % (bar, percent,iteration,total,txt), end='\r')

    if iteration >= total * 99/100:
        print()

def direct_load(path:str="D:/Data",sample:int=-1):
    images,labels,data = load_data(path,sample)
    pi = preprocess_images(images)
    pdi,pdl = preprocess_data(images,labels)
    bm = create_binary_masks(pdl)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(pi, bm)

    return x_train, x_val, x_test, y_train, y_val, y_test


def augment(x_train, x_val, x_test, y_train, y_val, y_test,batch:int=32):
    train_datagen = ImageDataGenerator(rescale=1./255)
    gen = train_datagen.flow(x_train, y_train, batch_size=batch, shuffle=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch, shuffle=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=batch, shuffle=True)
    return gen,val_generator,test_generator