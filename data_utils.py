import keras
import numpy as np
from keras import backend as K
import tensorflow as tf

#_BOUNDING_BOX = [37.639830, 37.929824, -123.173825, -122.281780] #sf
_BOUNDING_BOX = [40.477399,  40.917577, -74.259090, -73.700272] #ny

def geodistance_theano( p1 , p2 ):
  a0 = convertvalues(p1[:,0], _BOUNDING_BOX[0], _BOUNDING_BOX[1])
  a1 = convertvalues(p1[:,1], _BOUNDING_BOX[2],  _BOUNDING_BOX[3])
  b0 = convertvalues(p2[:,0], _BOUNDING_BOX[0] , _BOUNDING_BOX[1])
  b1 = convertvalues(p2[:,1], _BOUNDING_BOX[2],  _BOUNDING_BOX[3])

  aa0 = a0 * 3.141592653589793238462643383279502884197169399375105820974944592307816406286 / 180.0
  aa1 = a1 * 3.141592653589793238462643383279502884197169399375105820974944592307816406286 / 180.0
  bb0 = b0 * 3.141592653589793238462643383279502884197169399375105820974944592307816406286 / 180.0
  bb1 = b1 * 3.141592653589793238462643383279502884197169399375105820974944592307816406286 / 180.0

  sin_lat1 = K.sin( aa0 )
  cos_lat1 = K.cos( aa0 )
  sin_lat2 = K.sin( bb0 )
  cos_lat2 = K.cos( bb0 )
  delta_lng = bb1 - aa1
  cos_delta_lng = K.cos(delta_lng)
  sin_delta_lng = K.sin(delta_lng)
  d = tf.atan2(K.sqrt((cos_lat2 * sin_delta_lng) ** 2 + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2), sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng )
  return K.mean( 6371.0087714 * d , axis = -1 ) * 1000
  #return K.mean(d , axis = -1 )

def normalize_values (x, minv, maxv):
    return float(((float(x) - float(minv))/(float(maxv)-float(minv))))

def convertvalues(x, minv, maxv):
    return (x)*(maxv-minv)+minv
