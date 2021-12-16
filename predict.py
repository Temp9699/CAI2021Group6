import pickle
from collections import Counter
import numpy as np
from numpy.linalg import norm
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

training_img_labels = ["cold drink freezer", 
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "cold drink freezer",
                       "EDC", 
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC",
                        "EDC", 
                       "EDC",
                        "EDC",
                        "EDC", 
                       "EDC",
                        "EDC",
                       "ice maker", 
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",
                      "ice maker",]

#link file directory (hard code)
filenames = ["datasets/cai2/cold drink freezer/20211124_114324(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114329.jpg",
             "datasets/cai2/cold drink freezer/20211124_114333.jpg",
             "datasets/cai2/cold drink freezer/20211124_114334(0).jpg",
             "datasets/cai2/cold drink freezer/20211124_114340(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114343.jpg",
            "datasets/cai2/cold drink freezer/20211124_114331.jpg",
            "datasets/cai2/cold drink freezer/20211124_114358(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114404(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114344(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114342(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114342.jpg",
            "datasets/cai2/cold drink freezer/20211124_114347.jpg",
            "datasets/cai2/cold drink freezer/20211124_114348(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114352(0).jpg",
            "datasets/cai2/cold drink freezer/20211124_114359.jpg",
            "datasets/cai2/cold drink freezer/20211124_114407.jpg",
            "datasets/cai2/EDC/20211210_123243_001 - Copy.jpg",
             "datasets/cai2/EDC/20211210_123243_004.jpg",
             "datasets/cai2/EDC/20211210_123243_007.jpg",
             "datasets/cai2/EDC/20211210_123243_011.jpg",
             "datasets/cai2/EDC/20211210_123243_012.jpg",
             "datasets/cai2/EDC/20211210_123243_013.jpg",
             "datasets/cai2/EDC/20211210_123243_014.jpg",
             "datasets/cai2/EDC/20211210_123243_015.jpg",
             "datasets/cai2/EDC/20211210_123243_016.jpg",
             "datasets/cai2/EDC/20211210_123243_017.jpg",
             "datasets/cai2/EDC/20211210_123243_018.jpg",
             "datasets/cai2/EDC/20211210_123243_019.jpg",
             "datasets/cai2/EDC/20211210_123243_020.jpg",
             "datasets/cai2/EDC/20211210_123243_030.jpg",
             "datasets/cai2/EDC/20211210_123243_024.jpg",
             "datasets/cai2/EDC/20211210_123243_025.jpg",
             "datasets/cai2/EDC/20211210_123243_023.jpg",
             "datasets/cai2/EDC/20211210_123243_022.jpg",
             "datasets/cai2/EDC/20211210_123243_027.jpg",
             "datasets/cai2/EDC/20211210_123243_021.jpg",
             "datasets/cai2/EDC/20211210_123243_003.jpg",
             "datasets/cai2/EDC/20211210_123243_002.jpg",
             "datasets/cai2/EDC/20211210_123243_009.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122731_Burst01.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122740_Burst01.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst07.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst12.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst15.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst17.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122823_Burst19.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122936_Burst02.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122936_Burst05.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122936_Burst08.jpg",
             "datasets/cai2/ice maker/IMG_25641210_122936_Burst08.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst02.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst03.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst06.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst09.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst10.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst13.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst15.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst17.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst18.jpg",
             "datasets/cai2/ice maker/IMG_25641210_123026_Burst19.jpg",
            ]
def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path,
                         target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                        pooling='max')

#this is main module
import pickle
def predict(query_image):
    filenames = pickle.load(open('pickle/filenames-cai50.pickle', 'rb'))
    feature_list = pickle.load(open('pickle/features-cai50-resnet.pickle','rb'))
    class_ids = pickle.load(open('pickle/class_ids-cai50.pickle', 'rb'))
    #nearest neighbors fitting              
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)
    #add picture directory here           
    query_image_feature = extract_features(query_image ,model)
    distances, indices = neighbors.kneighbors([query_image_feature])

    similar_img_labels = [training_img_labels[similar_index] for similar_index in indices[0]]
    label_counter = Counter(similar_img_labels)
    most_common_labels =  label_counter.most_common(5)
    most_common_label_str = [label for label, freq in most_common_labels]
    indices = (indices)

    return most_common_label_str, indices[0]
