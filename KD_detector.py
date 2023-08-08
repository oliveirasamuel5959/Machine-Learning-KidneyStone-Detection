# ======================================
# Modules for reading the dataset file
import glob
import cv2
import os
import numpy as np

# ===================================================
# module used to divide the images into small patches
from patchify import patchify

# ===================================================
# Modules for feature extraction GLCM
from skimage.feature import greycomatrix, greycoprops
import pandas as pd
from tqdm import tqdm

# ===================================================
# Modules for training
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

import datetime
dt = datetime.datetime.now()
import warnings
warnings.filterwarnings(action='once')

kernel = np.array([
            [-1, 0, -1],
            [0, 8.7, -1],
            [-1, 0, -1]
])

def read_dataset(dataset, path, filter, size):
    train_images = []
    train_labels = []
    
    if dataset == 'train':
        path = path + "\Train\*"
        img_type = "*.png"
    else:
        path = path + "\Test\*"
        img_type = "*.png"
    
    for directory_path in glob.glob(path):
        label = directory_path.split("\\")[-1]
        print(label)

        for img_path in tqdm(glob.glob(os.path.join(directory_path, img_type))):
            img = cv2.imread(img_path, 0) #Reading color images
            img = cv2.resize(img, (254, 254) ) #Resize images
            #img = cv2.resize(img, (1038, 1328) ) #Resize images

            if filter == "filter2D":
                img = cv2.filter2D(img, -1, kernel)
            elif filter == "gaussian":
                img = cv2.GaussianBlur(img, (7, 7), 0)
            elif filter == "bilateral":
                img = cv2.bilateralFilter(img, 9, 75, 75)
            elif filter == "median":
                img = cv2.medianBlur(img, 5)
                    
            train_images.append(img)
            train_labels.append(label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    return train_images, train_labels


def images_patches(dataset, patchsize = 1):
    complete_image = []
    height = dataset[0].shape[0] / patchsize
    width = dataset[0].shape[1] / patchsize
    mean = (height + width) / 2
    for img in range(dataset.shape[0]):
        small_imgs = []
        large_image = dataset[img]
        
        patches = patchify(large_image, (int(height), int(width)), step= int(mean))
        
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch_img = patches[i,j,:,:]
                small_imgs.append(single_patch_img)
                
        #print("tamanho:", len(small_imgs))
        complete_image.append(small_imgs)
        
    complete_image = np.array(complete_image)
    
    return complete_image


def feature_extractor(dataset):
    image_list = []
    dict = {}

    image_dataset = pd.DataFrame()
    i = 1
    for j in range(dataset.shape[0]):
        for image in range(dataset[j].shape[0]):
            img = dataset[j][image]
            GLCM = greycomatrix(img, [5], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            #GLCM_Energy = graycoprops(GLCM, 'energy')[0, 0]
            #GLCM_corr = graycoprops(GLCM, 'correlation')[0, 0]
            GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0, 0]
            #GLCM_hom = graycoprops(GLCM, 'homogeneity')[0, 0]
            GLCM_contr = greycoprops(GLCM, 'contrast')[0, 0]
            #GLCM_asm = graycoprops(GLCM, 'ASM')[0, 0]
            
            '''dict.update({f"Energy{i}": [GLCM_Energy],
                    f"Corr{i}": [GLCM_corr],
                    f"Diss_sim{i}": [GLCM_diss],
                    f"Homogen{i}": [GLCM_hom],
                    f"ASM{i}": [GLCM_asm],
                    f"Contrast{i}": [GLCM_contr]})'''
            
            dict.update({f"Diss_sim{i}": [GLCM_diss],
                    f"Contrast{i}": [GLCM_contr]})
            
            i += 1
        df = pd.DataFrame(dict)
        image_dataset = image_dataset.append(df)
        dict.clear()
        i = 1 
    return image_dataset


def model_test(x_train, y_train, image_features, hidden_layer=(100,100), epochs=2000, activation='relu', learning_rate='constant', learning_rate_init=0, solver='sgd'):
    print("model training...")
    n_features = image_features.shape[1]
    image_features = np.expand_dims(image_features, axis=0)
    X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features
    print(X_for_ML.shape) #2d dimension

    mlp = MLPClassifier(
        activation=activation, 
        solver=solver, 
        learning_rate=learning_rate, 
        learning_rate_init=learning_rate_init, 
        hidden_layer_sizes=hidden_layer, 
        max_iter=epochs, 
        verbose=1)
    #mlp = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=2000, verbose=1) Result: accuracy = 0.62
    mlp.fit(X_for_ML, y_train)
    return mlp


def show_model_metrics(x_test, test_features, y_test):
    cm = confusion_matrix(y_test, test_prediction)
    
    fig, ax = plt.subplots(figsize=(6,6))    
    ax = sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
    ax.set_title('Confusion Matrix', size=25)
    ax.set_ylabel('Actual Values', size=20)
    ax.set_xlabel('Predicted Values', size=20)

def get_accuracy(y_test, test_prediction):
    return float(metrics.accuracy_score(y_test, test_prediction))

def get_sensibility(y_test, test_prediction):
    return float(metrics.recall_score(y_test, test_prediction))



def detector(path, filter='none', train_size=-1, test_size=-1, patchsize=1, hidden_layers=(100,100), epochs=2000):

    train_images, train_labels = read_dataset(dataset = 'train', path = path, filter=filter, size=train_size)

    le = preprocessing.LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    x_train, y_train = train_images, train_labels_encoded

    X_train = images_patches(x_train, patchsize=5)
    image_features = feature_extractor(X_train)
    
    train_mlp = model_test(x_train, y_train, image_features, hidden_layer=hidden_layers, epochs=epochs)

    test_images, test_labels = read_dataset(dataset = 'test', path=path, filter=filter, size=test_size)

    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    x_test, y_test = test_images, test_labels_encoded
    X_test = images_patches(x_test, patchsize=5)
    test_features = feature_extractor(X_test)

    test_features = np.expand_dims(test_features, axis=0)
    test_for_mlp = np.reshape(test_features, (x_test.shape[0], -1))
    test_prediction = train_mlp.predict(test_for_mlp)

    acc = get_accuracy(y_test, test_prediction)
    sensibility = get_sensibility(y_test, test_prediction)

    return acc, sensibility
