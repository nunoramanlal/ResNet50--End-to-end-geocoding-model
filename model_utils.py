from model_geocoding_utils_new import regression_model, classification_model, regre_class_model, regre_class_model2
from model_color_utils import y_true_max,  y_true_min, y_pred_max,  y_pred_min
from keras import models, layers, optimizers
import keras
from keras.layers import Conv2D, Input, Reshape, RepeatVector, concatenate, UpSampling2D, Flatten, Conv2DTranspose
from keras.models import Model
from keras import backend as K

img_size = 224

_COLOR_WEIGHTS = 'weights_mobilenet_model_color.h5'
_CLASS_WEIGHTS = ''
_REG_CLASS_WEIGHTS = 'mobilenet_ny_regclass_weights.h5'

def merged_model(flag, pesos=0, num_classes = 0):
    print('new model2')
    # encoder model
    encoder_ip = Input(shape=(img_size, img_size, 1))
    encoder1 = Conv2D(64, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder_ip)
    encoder = Conv2D(128, (3, 3), padding='same', activation='relu')(encoder1)
    encoder2 = Conv2D(128, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder)
    encoder = Conv2D(256, (3, 3), padding='same', activation='relu')(encoder2)
    encoder = Conv2D(256, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder)
    encoder = Conv2D(512, (3, 3), padding='same', activation='relu')(encoder)
    encoder = Conv2D(512, (3, 3), padding='same', activation='relu')(encoder)
    encoder = Conv2D(256, (3, 3), padding='same', activation='relu')(encoder)

    # input fusion
    # Decide the image shape at runtime to allow prediction on
    # any size image, even if training is on 128x128
    batch, height, width, channels = K.int_shape(encoder)

    feature_extraction_model = keras.applications.resnet50.ResNet50(
                                                        include_top=True,
                                                        weights='imagenet',
                                                        input_tensor=None,
                                                        input_shape=None,
                                                        pooling=None,
                                                        classes=1000)

    resnet_activations = Model(feature_extraction_model.input, feature_extraction_model.layers[-3].output)

    inp = Input(shape = (img_size,img_size,3))
    resnet_model_features = resnet_activations(inp)
    x = keras.layers.Conv2D(1000, (1, 1), padding='same', name='conv_preds')(resnet_model_features)
    a = Flatten()(x)


    fusion = RepeatVector(height * width)(a)
    fusion = Reshape((height, width, 1000))(fusion)
    fusion = concatenate([encoder, fusion], axis=-1)

    '''fusion = encoder'''
    fusion = Conv2D(256, (1, 1), padding='same', activation='relu')(fusion)

    # decoder model
    decoder = Conv2D(128, (3, 3), padding='same', activation='relu')(fusion)
    decoder = UpSampling2D()(decoder)
    decoder = concatenate([decoder, encoder2], axis=-1)
    decoder = Conv2D(64, (3, 3), padding='same', activation='relu')(decoder)
    decoder = Conv2D(64, (3, 3), padding='same', activation='relu')(decoder)
    decoder = UpSampling2D()(decoder)
    decoder = concatenate([decoder, encoder1], axis=-1)
    decoder = Conv2D(32, (3, 3), padding='same', activation='relu')(decoder)
    decoder = Conv2D(2, (3, 3), padding='same', activation='tanh')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    concat = concatenate([encoder_ip, decoder], axis = -1)

    color_model = Model([encoder_ip, inp], concat)
    color_model.load_weights(_COLOR_WEIGHTS)

    for layer in color_model.layers:
        layer.trainable=False

    if flag == 1:
        geo_model = regression_model()

    elif flag == 2:
        geo_model = classification_model(num_classes)
        #geo_model.load_weights(_CLASS_WEIGHTS)

    elif flag ==3:
        geo_model = regre_class_model(num_classes)
        geo_model.load_weights(_REG_CLASS_WEIGHTS)
    else:
        geo_model = regre_class_model2(num_classes, 820, pesos)

    merged_model = geo_model(concat)

    model = Model([encoder_ip, inp], merged_model)

    return model
