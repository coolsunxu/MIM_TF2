
import tensorflow as tf
from tensorflow.keras import layers # 导入常见网络层类

"""
EPSILON = 0.00001
def tensor_layer_norm(x, state_name):
	x_shape = x.get_shape() # 获得输入的大小
	dims = x_shape.ndims
	params_shape = x_shape[-1:]
	
	# 获取均值和方差
	if dims == 4:
		m, v = tf.nn.moments(x, [1,2,3], keepdims=True)
	elif dims == 5:
		m, v = tf.nn.moments(x, [1,2,3,4], keepdims=True)
	elif dims == 2:
		m, v = tf.nn.moments(x, [1], keepdims=True)
	else:
		raise ValueError('input tensor for layer normalization must be rank 4 or 5.')
	b = tf.Variable(initial_value=tf.zeros(params_shape),name = state_name+'b',trainable=False)
	s = tf.Variable(initial_value=tf.ones(params_shape),name = state_name+'s',trainable=False)
	x_tln = tf.nn.batch_normalization(x, m, v, b, s, EPSILON) # 归一化
	return x_tln
"""

# 返回归一化层
def tensor_layer_norm(state_name):
	return layers.BatchNormalization(name = state_name)


