from torch import nn

class SubSpectralNormalization(nn.Module):

	"""Subspectral normalization.
	"""
	def __init__(
		self,
		S,
		eps=1e-5
	):
		super().__init__()

		self.S = S
		self.eps = eps

	def forward(self, x):
		"""Forward pass of subspectral normalization

		Implementation taken from source paper:
		"SUBSPECTRAL NORMALIZATION FOR NEURAL AUDIO DATA PROCESSING"
		Simyung Chang, Hyoungwoo Park, Janghoon Cho, Hyunsin Park,
		Sungrack Yun, Kyuwoong Hwang.

		Note, we implement this following the authors' suggestion of using
		nn.BatchNorm2D(C * S)

		Args:
			x (torch.Tensor): features. shape = (batch, channels, frequency, time)
		"""
		N, C, F, T = x.size()
		x = x.view(N, C * self.S, F // self.S, T)

		batch_norm = nn.BatchNorm2d(C * self.S)
		x = batch_norm(x)

		return x.view(N, C, F, T)

class BCResNormalBlock(nn.Module):
	"""The normal block of broadcasted residual learning.
	This block is used in the network architecture when
	the preceeding block has the same number of output channels
	as this block has input channels. It has two residuals: an
	intermediate connection between the two parameterized functions
	f1 and f2 and the output of f1. It sums these two residuals with
	an identity connection, forming the output of the block.
	"""
	def __init__(
		self,
		input_channels = 8,
		output_channels = 8,
		stride = 1,
		dilation = 1,
		frequency_kernel_size = (3, 1),
		frequency_padding = (1, 0),
		temporal_kernel_size = (1, 3),
		temporal_padding = (0, 1),
	):
		super().__init__()

		assert input_channels == output_channels, 'For Normal Block, input channels must equal output channels. Otherwise, use a Transition Block.'

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
		self.subspectral_norm = SubSpectralNormalization(5)
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
	def __init__(
		self,
		input_channels = 16,
		output_channels = 8,
		stride = 1,
		dilation = 1,
		frequency_kernel_size = (3, 1),
		frequency_padding = (1, 0),
		temporal_kernel_size = (1, 3),
		temporal_padding = (0, 1),
	):
		super().__init__()

		assert input_channels != output_channels, 'For Transition Block, input channels must not equal output channels. Otherwise, use a Normal Block.'

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
		self.subspectral_norm = SubSpectralNormalization(5)
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

def build_stage(
		input_channels,
		output_channels,
		stride,
		dilation,
		number_of_normal_blocks,
		temporal_padding
	):
	"""Builds stages of residual blocks.

	Args:
		input_channels (int): Number of input channels at start of stage
		output_channels (int): Number of output channels at Transition block. Also, the number of input channels for the stage's normal blocks
		stride (int | tuple): The stride for the stage's transition block.
		dilation (int | tuple): The dilation for the stage's convolution layers
		number_of_normal_blocks (int): The number of normal blocks to include in the stage
		temporal_padding (tuple): The padding for the temporal depthwise convolutions

	Returns:
		nn.Sequential: An nn.Sequential module with the appropriate components for the stage described by the parameters.
	"""
	modules = []
	modules.append(
		BCResTransitionBlock(
			input_channels=input_channels,
			output_channels=output_channels,
			stride=stride,
			dilation=dilation,
			temporal_padding=temporal_padding
		),
	)

	for _ in range(number_of_normal_blocks):
		modules.append(
			BCResNormalBlock(
				input_channels=output_channels,
				output_channels=output_channels,
				stride=1,
				dilation=dilation,
				temporal_padding=temporal_padding
			),
		)
	
	return nn.Sequential(*modules)

class BroadcastedResNet(nn.Module):
	"""The BC-ResNet implementation of 

	"Broadcasted Residual Learning for Efficient Keyword Spotting"
	Byeonggeun Kim, Simyung Chang, Jinkyu Lee, Dooyong Sun
	https://www.isca-speech.org/archive/pdfs/interspeech_2021/kim21l_interspeech.pdf
	"""
	def __init__(
		self,
		stage_configurations = [
			{
				'input_channels': 16,
				'output_channels': 8,
				'stride': 1,
				'dilation': 1,
				'number_of_normal_blocks': 2,
				'temporal_padding': (0, 1),
			},
			{
				'input_channels': 8,
				'output_channels': 12,
				'stride': (2, 1),
				'dilation': (1, 2),
				'number_of_normal_blocks': 2,
				'temporal_padding': (0, 2)
			},
			{
				'input_channels': 12,
				'output_channels': 16,
				'stride': (2, 1),
				'dilation': (1, 4),
				'number_of_normal_blocks': 4,
				'temporal_padding': (0, 4)
			},
			{
				'input_channels': 16,
				'output_channels': 20,
				'stride': 1,
				'dilation': (1, 8),
				'number_of_normal_blocks': 4,
				'temporal_padding': (0, 8)
			}
		]
		):
		super().__init__()

		self.frontend_convolution = nn.Conv2d(
			in_channels=1,
			out_channels=16,
			stride=(2, 1),
			kernel_size=(5, 5),
			padding=(2, 2),
		)

		# NOTE: The changes in channel and stride belong to
		# the transition block. The dilation is given to both
		# the transition and normal blocks. Padding is calculated
		# for the output shape to match the details in the table.
		stages = []
		for config in stage_configurations:
			stages.append(build_stage(**config))

		self.stages = nn.Sequential(*stages)

		self.depthwise_convolution = nn.Conv2d(
			in_channels=20,
			out_channels=20,
			stride=1,
			kernel_size=(5, 1),
			dilation=1,
			groups=20
		)
		self.pointwise_convolution_1 = nn.Conv2d(
			in_channels=20,
			out_channels=32,
			kernel_size=(1, 1),
			stride=1,
			dilation=1,
		)
		self.pointwise_convolution_2 = nn.Conv2d(
			in_channels=32,
			out_channels=12,
			kernel_size=(1, 1)
		)

	def forward(self, x):
		"""The forward pass through BC-ResNet.

		Args:
			x (torch.Tensor): the TF representation. shape = (batch, channels, frequency, time)

		Returns:
			torch.Tensor: shape = (32, 12, 1, 1)
		"""
		x = self.frontend_convolution(x)

		x = self.stages(x)

		x = self.depthwise_convolution(x)
		x = self.pointwise_convolution_1(x)
		x = x.mean(dim=3, keepdim=True)
		x = self.pointwise_convolution_2(x)

		return x

def demo_model():
	import torch

	batch = 2
	input_channels = 1
	h = 40
	w = 100
	x = torch.rand((batch, input_channels, h, w))

	model = BroadcastedResNet()
	
	print('Input Shape: {}'.format(x.shape))
	y = model(x)
	print('Output Shape: {}'.format(y.shape))

if __name__ == "__main__":
	demo_model()
