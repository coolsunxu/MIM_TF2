
import tensorflow as tf

from models.mim import MIM

a = tf.random.normal((5,6,64,64,1))
b = tf.random.normal((5,2,64,64,1))

num_layers = 3
num_hidden = [64,64,64]
filter_size = 5
total_length = a.shape[1]
input_length = a.shape[1]
shape = a.get_shape().as_list()

stlstm = MIM(shape, num_layers, num_hidden, filter_size, total_length, input_length)
		 
new = stlstm(a,b)
print(new[0].shape)
print(new[1])
