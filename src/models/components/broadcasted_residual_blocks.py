from torch import nn

class BCResNormalBlock(nn.Module):
	"""The normal block of broadcasted residual learning.
	This block is used in the network architecture when
	the preceeding block has the same number of output channels
	as this block has input channels. It has two residuals: an
	intermediate connection between the two parameterized functions
	f1 and f2 and the output of f1. It sums these two residuals with
	an identity connection, forming the output of the block.
	"""
	def __init__(self, hparams: dict):
		super().__init__()

		input_channels = hparams['input_channels']
		output_channels = hparams['output_channels']
		assert input_channels == output_channels, 'For Normal Block, input channels must equal output channels. Otherwise, use a Transition Block.'
		
		stride = hparams['stride']
		dilation = hparams['dilation']
		frequency_kernel_size = hparams['frequency_kernel_size']
		temporal_kernel_size = hparams['temporal_kernel_size']
		frequency_padding = hparams['frequency_padding']
		temporal_padding = hparams['temporal_padding']

		#
		# Component parts
		#

		self.depthwise_frequency_convolution = nn.Conv2d(
			in_channels=input_channels,
			out_channels=output_channels,
			kernel_size=frequency_kernel_size,
			stride=stride,
			dilation=dilation,
			groups=input_channels,
			padding=frequency_padding,
		)
		self.subspectral_norm = lambda x: x # TODO: write this functionality.
		self.depthwise_temporal_convolution = nn.Conv2d(
			in_channels=input_channels,
			out_channels=output_channels,
			kernel_size=temporal_kernel_size,
			stride=stride,
			dilation=dilation,
			groups=input_channels,
			padding=temporal_padding
		)
		self.batch_norm = nn.BatchNorm2d(input_channels)
		self.swish_activation = nn.SiLU()
		self.pointwise_convolution = nn.Conv2d(
			in_channels=input_channels,
			out_channels=output_channels,
			kernel_size=(1, 1)
		)
		self.dropout = nn.Dropout2d(p=0.1)
		self.relu_activation = nn.ReLU()

	def forward(self, x):
		"""The forward pass of the Broadcasted Residual
		Normal Block. For details, see Figure 2 of

		"Broadcasted Residual Learning for Efficient Keyword Spotting"
		Byeonggeun Kim, Simyung Chang, Jinkyu Lee, Dooyong Sun
		https://www.isca-speech.org/archive/pdfs/interspeech_2021/kim21l_interspeech.pdf

		Args:
			x (torch.Tensor): shape is (batch x number_of_channels x frequency x time)
		"""
		identity_connection = x

		x_f2 = self.depthwise_frequency_convolution(x)
		x_f2 = self.subspectral_norm(x_f2)

		auxiliary_residual_connection = x_f2

		# Average over frequency dimension for Freq. Avg. Pool
		x = x_f2.mean(dim=2, keepdim=True) 

		x_f1 = self.depthwise_temporal_convolution(x)
		x_f1 = self.batch_norm(x_f1)
		x_f1 = self.swish_activation(x_f1)
		x_f1 = self.pointwise_convolution(x_f1)
		x_f1 = self.dropout(x_f1)

		# Note:
		# x_f1 is *not* the same shape as the other two summands here.
		# The broadcasting is taken care of automatically, there's no
		# need to repeat the matrix or anything like that. The broadcasting
		# is built in to torch.
		out = x_f1 + auxiliary_residual_connection + identity_connection
		y = self.relu_activation(out)

		return y

class BCResTransitionBlock(nn.Module):
	"""The transition block of broadcasted residual learning.
	This block is used in the network architecture when
	the preceeding block does not have the same number of output channels
	as this block has input channels. It foregoes the identity connection that
	the normal block uses. Similar to the normal block, the transition block 
	has two parameterized functions, f1 and f2. It sums the output of f1 with the
	output of f2, forming the output of the block. 
	"""
	def __init__(self, hparams: dict):
		super().__init__()

		input_channels = hparams['input_channels']
		output_channels = hparams['output_channels']
		assert input_channels != output_channels, 'For Transition Block, input channels must not equal output channels. Otherwise, use a Normal Block.'
		
		stride = hparams['stride']
		dilation = hparams['dilation']
		frequency_kernel_size = hparams['frequency_kernel_size']
		temporal_kernel_size = hparams['temporal_kernel_size']
		frequency_padding = hparams['frequency_padding']
		temporal_padding = hparams['temporal_padding']

		#
		# Component parts
		#

		# The 1 x 1 convolution handles changing the number of channels to 
		# the output number of channels for the block. Every other subsequent
		# component uses the "output channels" as their input channels. 
		self.pointwise_convolution_for_channel_change = nn.Conv2d(
			in_channels=input_channels,
			out_channels=output_channels,
			kernel_size=(1, 1)
		)
		self.batch_norm_f2 = nn.BatchNorm2d(output_channels)
		self.relu_activation_f2 = nn.ReLU()
		self.depthwise_frequency_convolution = nn.Conv2d(
			in_channels=output_channels,
			out_channels=output_channels,
			kernel_size=frequency_kernel_size,
			stride=stride,
			dilation=dilation,
			groups=output_channels,
			padding=frequency_padding,
		)
		self.subspectral_norm = lambda x: x # TODO: write this functionality.
		self.depthwise_temporal_convolution = nn.Conv2d(
			in_channels=output_channels,
			out_channels=output_channels,
			kernel_size=temporal_kernel_size,
			stride=stride,
			dilation=dilation,
			groups=output_channels,
			padding=temporal_padding
		)
		self.batch_norm_f1 = nn.BatchNorm2d(output_channels)
		self.swish_activation = nn.SiLU()
		self.pointwise_convolution = nn.Conv2d(
			in_channels=output_channels,
			out_channels=output_channels,
			kernel_size=(1, 1)
		)
		self.dropout = nn.Dropout2d(p=0.1)
		self.relu_activation_out = nn.ReLU()

	def forward(self, x):
		"""The forward pass of the Broadcasted Residual
		Transition Block. For details, see Figure 2 of

		"Broadcasted Residual Learning for Efficient Keyword Spotting"
		Byeonggeun Kim, Simyung Chang, Jinkyu Lee, Dooyong Sun
		https://www.isca-speech.org/archive/pdfs/interspeech_2021/kim21l_interspeech.pdf

		Args:
			x (torch.Tensor): shape is (batch x number_of_channels x frequency x time)
		"""
		x_f2 = self.pointwise_convolution_for_channel_change(x)
		x_f2 = self.batch_norm_f2(x_f2)
		x_f2 = self.relu_activation_f2(x_f2)
		x_f2 = self.depthwise_frequency_convolution(x_f2)
		x_f2 = self.subspectral_norm(x_f2)		

		auxiliary_residual_connection = x_f2

		# Average over frequency dimension for Freq. Avg. Pool
		x = x_f2.mean(dim=2, keepdim=True)

		x_f1 = self.depthwise_temporal_convolution(x)
		x_f1 = self.batch_norm_f1(x_f1)
		x_f1 = self.swish_activation(x_f1)
		x_f1 = self.pointwise_convolution(x_f1)
		x_f1 = self.dropout(x_f1)

		# Note:
		# x_f1 is *not* the same shape as the other summand here.
		# The broadcasting is taken care of automatically, there's no
		# need to repeat the matrix or anything like that. The broadcasting
		# is built in to torch.
		out = x_f1 + auxiliary_residual_connection
		y = self.relu_activation_out(out)

		return y

def debug_normal_block():
	import torch

	batch = 2
	channels = 1
	h = 40
	w = 100
	x = torch.rand((batch, channels, h, w))
	hparams = {
		'input_channels': channels,
		'output_channels': channels,
		'stride': 1,
		'dilation': 1,
		'frequency_kernel_size': (3, 1),
		'frequency_padding': (1, 0),
		'temporal_kernel_size': (1, 3),
		'temporal_padding': (0, 1),
	}

	normal_res_block = BCResNormalBlock(hparams)
	y = normal_res_block(x)
	
	print('debugging normal block')

def debug_transition_block():
	import torch

	batch = 2
	input_channels = 1
	output_channels = 4
	h = 40
	w = 100
	x = torch.rand((batch, input_channels, h, w))
	hparams = {
		'input_channels': input_channels,
		'output_channels': output_channels,
		'stride': 1,
		'dilation': 1,
		'frequency_kernel_size': (3, 1),
		'frequency_padding': (1, 0),
		'temporal_kernel_size': (1, 3),
		'temporal_padding': (0, 1),
	}

	transition_res_block = BCResTransitionBlock(hparams)
	y = transition_res_block(x)
	
	print('debugging transition block')


if __name__ == "__main__":
	# debug_normal_block()
	debug_transition_block()