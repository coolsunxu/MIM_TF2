
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # 导入常见网络层类
import sys
sys.path.append("..")
from layers.TensorLayerNorm import tensor_layer_norm
import math


class MIMBlock(keras.Model):
	def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
				 seq_shape, x_shape_in, tln=False, initializer=None):
		super(MIMBlock, self).__init__()
		
		"""Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			forget_bias: float, The bias added to forget gates (see above).
			tln: whether to apply tensor layer normalization
		"""
		self.layer_name = layer_name # 当前网络层名
		self.filter_size = filter_size # 卷积核大小
		self.num_hidden_in = num_hidden_in # 隐藏层输入
		self.num_hidden = num_hidden # 隐藏层大小
		self.convlstm_c = None # 
		self.batch = seq_shape[0] # batch_size
		self.height = seq_shape[2] # 图片高度
		self.width = seq_shape[3] # 图片宽度
		self.x_shape_in = x_shape_in # 通道数
		self.layer_norm = tln # 是否归一化
		self._forget_bias = 1.0 # 遗忘参数

		def w_initializer(dim_in, dim_out):
			random_range = math.sqrt(6.0 / (dim_in + dim_out))
			return tf.random_uniform_initializer(-random_range, random_range)
		if initializer is None or initializer == -1: # 初始化参数
			self.initializer = w_initializer
		else:
			self.initializer = tf.random_uniform_initializer(-initializer, initializer)
			
		# MIMS
		
		# h_t
		self.mims_h_t = layers.Conv2D(self.num_hidden * 4,
					self.filter_size, 1, padding='same',
					kernel_initializer=self.initializer(self.num_hidden, self.num_hidden * 4),
					name='state_to_state')
					
		# c_t
		self.ct_weight = tf.Variable(initial_value = tf.random.normal(shape=[self.height,self.width,self.num_hidden*2],
					mean=0,stddev=1),name = 'c_t_weight',trainable=True)

		# x
		self.mims_x = layers.Conv2D(self.num_hidden * 4,
					self.filter_size, 1,
					padding='same',
					kernel_initializer=self.initializer(self.num_hidden, self.num_hidden * 4),
					name='input_to_state')
					
		# oc
		self.oc_weight = tf.Variable(initial_value = tf.random.normal(shape=[self.height,self.width,self.num_hidden],
					mean=0,stddev=1),name = 'oc_weight',trainable=True)
					
		# MIMBLOCK	
		# h
		self.t_cc = layers.Conv2D(
				self.num_hidden*3, # 网络输入 输出通道数
				self.filter_size, 1, padding='same', # 滤波器大小 步长 填充方式
				kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden*3), # 参数初始化
				name='time_state_to_state')
				
		# m
		self.s_cc = layers.Conv2D(
				self.num_hidden*4,  # 网络输入 输出通道数
				self.filter_size, 1, padding='same', # 滤波器大小 步长 填充方式
				kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden*4),
				name='spatio_state_to_state')
				
		# x
		self.x_cc = layers.Conv2D(
				self.num_hidden*4, # 网络输入 输出通道数
				self.filter_size, 1, padding='same', # 滤波器大小 步长 填充方式
				kernel_initializer=self.initializer(self.x_shape_in, self.num_hidden*4), # 参数初始化
				name='input_to_state')
		
		# c 
		self.c_cc = layers.Conv2D(
				self.num_hidden,  # 网络输入 输出通道数
				1, 1, padding='same', # 滤波器大小 步长 填充方式
				name='cell_reduce')
					
	def init_state(self): # 初始化lstm 隐藏层状态
		return tf.zeros([self.batch, self.height, self.width, self.num_hidden],
						dtype=tf.float32)

	def MIMS(self, x, h_t, c_t): # MIMS
		
		# h_t c_t[batch, in_height, in_width, num_hidden]
		# 初始化隐藏层 记忆 空间
		
		if h_t is None:
			h_t = self.init_state()
		if c_t is None:
			c_t = self.init_state()
			
		# h_t
		h_concat = self.mims_h_t(h_t)
		
		if self.layer_norm: # 是否归一化
			h_concat = tensor_layer_norm(h_concat, 'state_to_state')
		
		# 在第3维度上切分为4份 因为隐藏层是4*num_hidden 
		i_h, g_h, f_h, o_h = tf.split(h_concat, 4, 3)

		# ct_weight
		ct_activation = tf.multiply(tf.tile(c_t, [1,1,1,2]), self.ct_weight)
		i_c, f_c = tf.split(ct_activation, 2, 3)

		i_ = i_h + i_c
		f_ = f_h + f_c
		g_ = g_h
		o_ = o_h

		if x is not None:
			# x 
			x_concat = self.mims_x(x)
			
			if self.layer_norm:
				x_concat = tensor_layer_norm(x_concat, 'input_to_state')
			i_x, g_x, f_x, o_x = tf.split(x_concat, 4, 3)

			i_ += i_x
			f_ += f_x
			g_ += g_x
			o_ += o_x

		i_ = tf.nn.sigmoid(i_)
		f_ = tf.nn.sigmoid(f_ + self._forget_bias)
		c_new = f_ * c_t + i_ * tf.nn.tanh(g_)

		# oc_weight
		o_c = tf.multiply(c_new, self.oc_weight)
		
		h_new = tf.nn.sigmoid(o_ + o_c) * tf.nn.tanh(c_new)

		return h_new, c_new

	def call(self, x, diff_h, h, c, m):
		
		# 初始化隐藏层 记忆 空间
		
		if h is None:
			h = self.init_state()
		if c is None:
			c = self.init_state()
		if m is None:
			m = self.init_state()
		if diff_h is None:
			diff_h = tf.zeros_like(h)
			
		# h
		t_cc = self.t_cc(h)
		
		# m
		s_cc = self.s_cc(m)
			
		# x
		x_cc = self.x_cc(x)
			
		if self.layer_norm:
			t_cc = tensor_layer_norm(t_cc, 'time_state_to_state')
			s_cc = tensor_layer_norm(s_cc, 'spatio_state_to_state')
			x_cc = tensor_layer_norm(x_cc, 'input_to_state')

		i_s, g_s, f_s, o_s = tf.split(s_cc, 4, 3)
		i_t, g_t, o_t = tf.split(t_cc, 3, 3)
		i_x, g_x, f_x, o_x = tf.split(x_cc, 4, 3)

		i = tf.nn.sigmoid(i_x + i_t)
		i_ = tf.nn.sigmoid(i_x + i_s)
		g = tf.nn.tanh(g_x + g_t)
		g_ = tf.nn.tanh(g_x + g_s)
		f_ = tf.nn.sigmoid(f_x + f_s + self._forget_bias)
		o = tf.nn.sigmoid(o_x + o_t + o_s)
		new_m = f_ * m + i_ * g_
		
		# MIMS
		c, self.convlstm_c = self.MIMS(diff_h, c, self.convlstm_c)
		
		new_c = c + i * g
		cell = tf.concat([new_c, new_m], 3)
		
		# c
		cell = self.c_cc(cell)
								
		new_h = o * tf.nn.tanh(cell)

		return new_h, new_c, new_m # 大小均为 [batch, in_height, in_width, num_hidden]
		
if __name__ == '__main__':
	a = tf.random.normal((32,64,64,1))
	b = tf.random.normal((32,64,64,64))
	layer_name = 'stlstm'
	filter_size = 5
	num_hidden_in = 64
	num_hidden = 64
	seq_shape = [32,12,64,64,1]
	x_shape_in = 1
	tln = True
	
	stlstm = MIMBlock(layer_name, filter_size, num_hidden_in, num_hidden,
				 seq_shape, x_shape_in, tln)
	
			 
	new_h, new_c, new_m = stlstm(a,b,None,None,None)
	print(new_h.shape)
	print(new_c.shape)
	print(new_m.shape)
	










