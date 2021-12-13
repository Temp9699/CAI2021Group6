import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import os
import random
import time
import math
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
import pickle

def predict(input_image):

    filenames = pickle.load(open('pickle/filenames-cai50.pickle', 'rb'))
    feature_list = pickle.load(open('pickle/features-cai50-resnet.pickle','rb'))
    class_ids = pickle.load(open('pickle/class_ids-cai50.pickle', 'rb'))
                        
    #nearest neighbors fitting             
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)
    #add picture directory here                                                
    query_image_feature = extract_features('datasets/cai2/cold drink freezer/20211124_114352.jpg' ,model)
    distances, indices = neighbors.kneighbors([query_image_feature])

    similar_img_labels = [training_img_labels[similar_index] for similar_index in indices[0]]
    label_counter = Counter(similar_img_labels)
    most_common_labels =  label_counter.most_common(5)
    most_common_label_str = [label for label, freq in most_common_labels]

    result = "Possible labels are " + most_common_label_str
    return result