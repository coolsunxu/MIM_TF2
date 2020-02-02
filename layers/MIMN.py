
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # 导入常见网络层类
import sys
sys.path.append("..")
from layers.TensorLayerNorm import tensor_layer_norm

class MIMN(keras.Model):
	def __init__(self, layer_name, filter_size, num_hidden, seq_shape, tln=True, initializer=0.001):
		super(MIMN, self).__init__()
		"""Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			tln: whether to apply tensor layer normalization.
		"""
		self.layer_name = layer_name # 当前网络层名
		self.filter_size = filter_size # 卷积核大小
		self.num_hidden = num_hidden # 隐藏层大小
		self.layer_norm = tln # 是否归一化
		self.batch = seq_shape[0] # batch_size
		self.height = seq_shape[2] # 图片高度
		self.width = seq_shape[3] # 图片宽度
		self._forget_bias = 1.0 # 遗忘参数
		if initializer == -1: # 初始化参数
			self.initializer = None
		else:
			self.initializer = tf.random_uniform_initializer(-initializer,initializer)
			
		# h_t
		self.h_t = layers.Conv2D(self.num_hidden * 4,
					self.filter_size, 1, padding='same',
					kernel_initializer=self.initializer,
					name='state_to_state')
					
		# c_t
		self.ct_weight = tf.Variable(initial_value = tf.random.normal(shape=[self.height,self.width,self.num_hidden*2],
					mean=0,stddev=1),name = 'c_t_weight',trainable=True)

		# x
		self.x = layers.Conv2D(self.num_hidden * 4,
					self.filter_size, 1,
					padding='same',
					kernel_initializer=self.initializer,
					name='input_to_state')
					
		# oc
		self.oc_weight = tf.Variable(initial_value = tf.random.normal(shape=[self.height,self.width,self.num_hidden],
					mean=0,stddev=1),name = 'oc_weight',trainable=True)
					
		# bn 
		self.bn_h_concat = tensor_layer_norm('mimn_state_to_state')
		self.bn_x_concat = tensor_layer_norm('mimn_input_to_state')

	def init_state(self): # 初始化lstm 隐藏层状态
		shape = [self.batch, self.height, self.width, self.num_hidden]
		return tf.zeros(shape, dtype=tf.float32)

	def call(self, x, h_t, c_t):
		
		# h c [batch, in_height, in_width, num_hidden]
		
		# 初始化隐藏层 记忆 空间
		
		if h_t is None:
			h_t = self.init_state()
		if c_t is None:
			c_t = self.init_state()
		
		# 1
		h_concat = self.h_t(h_t)
		
		if self.layer_norm:
			h_concat = self.bn_h_concat(h_concat)
		i_h, g_h, f_h, o_h = tf.split(h_concat, 4, 3)
		
		# 2 变量 可训练
		ct_activation = tf.multiply(tf.tile(c_t, [1,1,1,2]), self.ct_weight)
		i_c, f_c = tf.split(ct_activation, 2, 3)

		i_ = i_h + i_c
		f_ = f_h + f_c
		g_ = g_h
		o_ = o_h

		if x is not None:
			# 3 x
			x_concat = self.x(x)
			
			if self.layer_norm:
				x_concat = self.bn_x_concat(x_concat)
			i_x, g_x, f_x, o_x = tf.split(x_concat, 4, 3)

			i_ += i_x
			f_ += f_x
			g_ += g_x
			o_ += o_x

		i_ = tf.nn.sigmoid(i_)
		f_ = tf.nn.sigmoid(f_ + self._forget_bias)
		c_new = f_ * c_t + i_ * tf.nn.tanh(g_)

		# 4 变量 可训练
		o_c = tf.multiply(c_new, self.oc_weight)

		h_new = tf.nn.sigmoid(o_ + o_c) * tf.nn.tanh(c_new)

		return h_new, c_new # 大小均为 [batch, in_height, in_width, num_hidden]
		
if __name__ == '__main__':
	a = tf.random.normal((32,64,64,1))
	layer_name = 'stlstm'
	filter_size = 5
	num_hidden_in = 64
	num_hidden = 64
	seq_shape = [32,12,64,64,1]
	tln = True
	
	stlstm = MIMN(layer_name, filter_size, num_hidden,
				 seq_shape, tln)
	
	new_h, new_c = stlstm(a,None,None)
	print(new_h.shape)
	print(new_c.shape)

