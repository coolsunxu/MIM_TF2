
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # 导入常见网络层类

import sys
sys.path.append("..")
from layers.TensorLayerNorm import tensor_layer_norm

class SpatioTemporalLSTMCell(keras.Model): # stlstm 
	def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
				 seq_shape, x_shape_in, tln=False, initializer=None):
		super(SpatioTemporalLSTMCell, self).__init__()
		
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
		self.num_hidden_in = num_hidden_in # 隐藏层输入大小
		self.num_hidden = num_hidden # 隐藏层数量
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
			
		# 建立网络层
		# h
		self.t_cc = layers.Conv2D(
				self.num_hidden*4, # 网络输入 输出通道数
				self.filter_size, 1, padding='same', # 滤波器大小 步长 填充方式
				kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden*4), # 参数初始化
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
				kernel_initializer=self.initializer(self.num_hidden*2, self.num_hidden), # 参数初始化
				name='cell_reduce')
				
		# bn
		self.bn_t_cc = tensor_layer_norm('st_time_state_to_state')
		self.bn_s_cc = tensor_layer_norm('st_spatio_state_to_state')
		self.bn_x_cc = tensor_layer_norm('st_input_to_state')
				
	def init_state(self): # 初始化lstm 隐藏层状态
		return tf.zeros([self.batch, self.height, self.width, self.num_hidden],
						dtype=tf.float32)

	def call(self, x, h, c, m):
		
		# x [batch, in_height, in_width, in_channels]
		# h c m [batch, in_height, in_width, num_hidden]
		
		# 初始化隐藏层 记忆 空间
		if h is None:
			h = self.init_state()
		if c is None:
			c = self.init_state()
		if m is None:
			m = self.init_state()
		
		# 计算网络输出
		t_cc = self.t_cc(h)
		s_cc = self.s_cc(m)
		x_cc = self.x_cc(x)
		
		if self.layer_norm:
			# 计算均值 标准差 归一化
			t_cc = self.bn_t_cc(t_cc)
			s_cc = self.bn_s_cc(s_cc)
			x_cc = self.bn_x_cc(x_cc)
		
		# 在第3维度上切分为4份 因为隐藏层是4*num_hidden 
		i_s, g_s, f_s, o_s = tf.split(s_cc, 4, 3) # [batch, in_height, in_width, num_hidden]
		i_t, g_t, f_t, o_t = tf.split(t_cc, 4, 3)
		i_x, g_x, f_x, o_x = tf.split(x_cc, 4, 3)

		i = tf.nn.sigmoid(i_x + i_t)
		i_ = tf.nn.sigmoid(i_x + i_s)
		g = tf.nn.tanh(g_x + g_t)
		g_ = tf.nn.tanh(g_x + g_s)
		f = tf.nn.sigmoid(f_x + f_t + self._forget_bias)
		f_ = tf.nn.sigmoid(f_x + f_s + self._forget_bias)
		o = tf.nn.sigmoid(o_x + o_t + o_s)
		new_m = f_ * m + i_ * g_
		new_c = f * c + i * g
		cell = tf.concat([new_c, new_m],3) # [batch, in_height, in_width, 2*num_hidden]
		
		cell = self.c_cc(cell)
		
		new_h = o * tf.nn.tanh(cell)

		return new_h, new_c, new_m # 大小均为 [batch, in_height, in_width, num_hidden]
		
if __name__ == '__main__':
	a = tf.random.normal((32,64,64,1))
	layer_name = 'stlstm'
	filter_size = 5
	num_hidden_in = 64
	num_hidden = 64
	seq_shape = [32,12,64,64,1]
	x_shape_in = 1
	tln = True
	
	stlstm = SpatioTemporalLSTMCell(layer_name, filter_size, num_hidden_in, num_hidden,
				 seq_shape, x_shape_in, tln)
	"""			 
	model = keras.Sequential()
	model.add(stlstm)
	"""
			 
	new_h, new_c, new_m = stlstm(a,None,None,None)
	print(new_h.shape)
	print(new_c.shape)
	print(new_m.shape)
	









