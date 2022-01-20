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