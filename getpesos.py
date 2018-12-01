from keras.applications import resnet50
from keras import models, layers, optimizers
import numpy as np
import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import keras
import os, sys
import numpy as np
import collections
import tensorflow as tf
import _healpy, math, healpy, pickle, sys
from data_utils import geodistance_theano, normalize_values

_REG_CLASS_WEIGHTS = 'mobilenet_ny_regclass_weights.h5'
_TRAIN_TEST_SPLIT_SEED = 2
_IMAGES1 = '../old_resized_images_ny/'
_IMAGES2 = '../resized_images_ny2/'

def regre_class_model2(num_classes):
    #Load the MobileNet model
    image_size=224
    mobilenet_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3), pooling = 'avg')
    #mobilenet_model.summary()

    #build model
    inp = layers.Input(shape=(224, 224, 3))
    x = mobilenet_model (inp)
    z = layers.Dense(2, activation = 'sigmoid')(x)
    layer = layers.Dense(num_classes, activation = 'softmax')
    q = layer(x)

    model = models.Model(inp, [z,q])
    model.load_weights(_REG_CLASS_WEIGHTS)

    return layer.get_weights()

def get_images(path):
    images_list = os.listdir(path) #list of all images
    images = []
    coordinates = []
    for line in images_list:
        images.append(line)
        entry = os.path.splitext(line)[0].split(",") #filename without the extension
        coordinates.append((float(entry[2].rstrip()), float(entry[1]))) #(latitude, longitude)
    return images, coordinates

def get_array(training_images, training_coordinates, tst, num_img):
    X_train, X_test, Y_train, Y_test = train_test_split(training_images, training_coordinates,
                                        test_size=tst, random_state = _TRAIN_TEST_SPLIT_SEED)
    encoder = LabelEncoder()
    Y_train = [ _healpy.latlon2healpix( i[0] , i[1] , math.pow(4 , 6) ) for i in Y_train ]
    Y_test = [ _healpy.latlon2healpix( i[0] , i[1] , math.pow(4 , 6) ) for i in Y_test ]
    fit_trans = encoder.fit_transform( Y_test+Y_train)
    _encoder = np_utils.to_categorical(fit_trans)
    _newenconder = _encoder.astype(int)
    i = 0
    dict = {}
    while i!=num_img:
        index = np.argmax(_newenconder[i])
        _class = encoder.classes_[index]
        i+=1
        dict[_class] = index # a classe x esta representada no index z

    dict = collections.OrderedDict(sorted(dict.items()))
    return dict


def _main():
    a, b = get_images(_IMAGES1)
    old_classes = get_array(a, b, 0.50, 37618)

    a, b = get_images(_IMAGES2)
    flickr_classes = get_array(a, b, 0.20, 1139379)

    dict3 = {}
    for _class in old_classes.keys():
        if not(_class in flickr_classes.keys()):
            dict3[_class] = 'nan'
        else:
            dict3[_class] = flickr_classes[_class] #key: class vlue_index

    #print ('.......................done...........................')
    pesos = regre_class_model2(820) #ja tenho aqui os pesos
    #print(np.array(pesos).shape)
    #print (np.array(pesos[0]).shape) #shape(1024,427)
    #print (np.array(pesos[1]).shape) #shape(,427)

    values = list(dict3.values())
    #get bias
    bias = pesos[1]
    _new_bias = []
    for index in values:
        if(str(index) == 'nan'):
            _new_bias.append(-156.4698)
        else:
            _new_bias.append(bias[int(index)])

    #get ?
    _a = pesos[0]
    _new_a = []
    for _list in _a:
        _temp_a = []
        for index in values:
            if(str(index) == 'nan'):
                _temp_a.append(_list[0])
            else:
                _temp_a.append(_list[index])
        _new_a.append(_temp_a)

    _new_a = np.array(_new_a, dtype = np.float32)
    _new_bias = np.array(_new_bias, dtype = np.float32)
    print(_new_a.shape)
    print(_new_bias.shape)
    new_pesos = [_new_a, _new_bias]

    return new_pesos
