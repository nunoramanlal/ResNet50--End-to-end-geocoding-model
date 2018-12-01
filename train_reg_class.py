import keras
from sklearn.preprocessing import LabelEncoder
import _healpy, math, healpy, pickle, keras, sys
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from skimage.io import imsave, imread
from keras.applications.mobilenet import preprocess_input
from data_utils import geodistance_theano, normalize_values
from model_utils import merged_model
import tensorflow as tf
import numpy as np
import random as rn
import os
from getpesos import _main
from predict_class_reg import e, f


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))

#_BOUNDING_BOX = [37.639830, 37.929824, -123.173825, -122.281780] #sf
_BOUNDING_BOX = [40.477399,  40.917577, -74.259090, -73.700272] #ny

_IMAGES = '../old_resized_images_ny/'
_MODEL_FINAL_NAME  = 'pret_reg_class_color_ny(fold1).h5'
_MODEL_WEIGHTS_FINAL_NAME = 'pret_reg_class_color_ny_weights(fold1).h5'
_TRAIN_TEST_SPLIT_SEED = 2
_X_TEST_FILE = 'pret_x_test_color_reg_class_ny.txt'
_X_TRAIN_FILE = 'pret_x_train_color_reg_class_ny.txt'
_ENCODER  = 'pret_enconder_color_reg_class_ny(fold1).p'

def get_images(path):
	images_list = os.listdir(path)#list of all images
	images = []
	coordinates = []
	for line in images_list:
		images.append(line)
		entry = os.path.splitext(line)[0].split(",") #filename without the extension
		coordinates.append((float(entry[2].rstrip()), float(entry[1])))
	return images, coordinates

def generator(X, Y, classes_, img_path, batch_size):
	while 1:
		line = -1
		new_X1 = np.zeros((batch_size, 224, 224, 1))
		new_X2 = np.zeros((batch_size, 224, 224, 3))
		new_Y2 = np.zeros((batchsize, len(classes_[0])))
		new_Y1 = np.zeros((batchsize, 2))
		count = 0
		for entry in X:
			if count < batch_size:
				line+=1
				#load img
				x_b = imread(img_path + str(entry))
				x_b = np.expand_dims(x_b, axis=0)
				x_b = np.array(x_b)

				#define input
				grayscaled_rgb = gray2rgb(rgb2gray(x_b))  # convert to 3 channeled grayscale image
				grayscaled_rgb = np.array(grayscaled_rgb)*255
				grayscaled_rgb = preprocess_input(grayscaled_rgb)
				lab_batch = rgb2lab(x_b)  # convert to LAB colorspace #usar o grayscaled_rgb
				X_batch = lab_batch[:, :, :, 0]  # extract L from LAB
				X_batch = X_batch.reshape(X_batch.shape + (1,))  # reshape into (batch, IMAGE_SIZE, IMAGE_SIZE, 1)
				X_batch = 2 * X_batch / 100 - 1.  # normalize the batch

				#define target
				a = normalize_values(Y[line][0], _BOUNDING_BOX[0], _BOUNDING_BOX[1] )
				b = normalize_values(Y[line][1], _BOUNDING_BOX[2], _BOUNDING_BOX[3])
				y = [(float(a), float(b))]

				z = [classes_[line]]

				new_X1[count,:] = X_batch
				new_X2[count,:] = grayscaled_rgb
				new_Y1[count,:] = np.array(y)
				new_Y2[count,:] = np.array(z)
				count+=1
			else:
				yield ([new_X1, new_X2], [new_Y1,new_Y2])
				count = 0
				new_X1 = np.zeros((batch_size, 224, 224, 1))
				new_x2 = np.zeros((batch_size, 224, 224, 3))
				new_Y1 = np.zeros((batch_size, 2))
				new_Y2 = np.zeros((batchsize, len(classes_[0])))

		if(np.count_nonzero(new_X1) != 0):
				yield ([new_X1, new_X2], [new_Y1,new_Y2])


training_images, training_coordinates = get_images(_IMAGES)
X_train, X_test, Y_train, Y_test = train_test_split(training_images, training_coordinates,
									test_size=0.50, random_state = _TRAIN_TEST_SPLIT_SEED)
X_train_size = len(X_train)

toWrite = ''
for inst in X_test:
	toWrite += inst
	toWrite += '\n'

file = open(_X_TEST_FILE, 'w')
file.write(str(toWrite))
file.close()

toWrite = ''
for inst in X_train:
	toWrite += inst
	toWrite += '\n'

file = open(_X_TRAIN_FILE, 'w')
file.write(str(toWrite))
file.close()

encoder = LabelEncoder()
Y_train_class = [ _healpy.latlon2healpix( i[0] , i[1] , math.pow(4 , 6) ) for i in Y_train ]
Y_test_class = [ _healpy.latlon2healpix( i[0] , i[1] , math.pow(4 , 6) ) for i in Y_test ]
fit_trans = encoder.fit_transform( Y_train_class + Y_test_class )
_encoder = np_utils.to_categorical(fit_trans)
_newenconder = _encoder.astype(int)
_NUMCLASSES = len(_newenconder[0])
print('NUM OF CLASSES --->', _NUMCLASSES )
Y_train_class = _newenconder[:-len(Y_test_class)]
pickle.dump(encoder, open(_ENCODER, 'wb'))
pesos = _main()
model = merged_model(4, pesos, _NUMCLASSES)
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(optimizer=opt, loss = [geodistance_theano,'categorical_crossentropy'])

batchsize = 32
earlyStopping=keras.callbacks.EarlyStopping(monitor = 'loss', patience=2)
checkpoint = ModelCheckpoint(_MODEL_WEIGHTS_FINAL_NAME, monitor='loss',
							 save_best_only=False, save_weights_only=True)
history = model.fit_generator(generator(X_train, Y_train, Y_train_class, _IMAGES, batchsize),
							epochs=100,
							steps_per_epoch= X_train_size/batchsize,
							callbacks=[earlyStopping, checkpoint])

model.save(_MODEL_FINAL_NAME)
model.save_weights(_MODEL_WEIGHTS_FINAL_NAME)
