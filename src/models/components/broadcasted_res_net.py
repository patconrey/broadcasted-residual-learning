from torch import nn
from broadcasted_residual_blocks import build_stage

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

def debug_model():
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
	debug_model()

