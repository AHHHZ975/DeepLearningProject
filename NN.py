from torch.nn import Sequential
from torch.nn import Module         # Rather than using the Sequential PyTorch class to implement LeNet, we’ll instead subclass the Module object so you can see how PyTorch implements neural networks using classes
from torch.nn import Conv2d         # PyTorch’s implementation of convolutional layers
from torch.nn import MultiheadAttention        # PyTorch’s implementation of convolutional layers
from torch.nn import Linear         # Fully connected layers
from torch.nn import LayerNorm
from torch.nn import Dropout
from torch.nn import MaxPool2d      # Applies 2D max-pooling to reduce the spatial dimensions of the input volume
from torch.nn import ReLU           # ReLU activation function
from torch.nn import GELU
from torch.nn import Tanh
import sys
import numpy as np
from torch.nn.modules.activation import Tanh
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.flatten import Flatten

from Utils.Utils import imageToPatches
sys.path.append('C:/Users/AmirHossein/OneDrive/Desktop/DeepLearningProject/DeepLearningProject')
import Config as cfg


# Creating a PyTorch class
# 128*128 ==> 16 ==> 1000*3
class AE(Module):
	def __init__(self):
		super(AE, self).__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 784 ==> 9
		self.encoder = Sequential(
			# Linear(cfg.IMAGE_SIZE*3, 96*96),
			# ReLU(),
			# Linear(96*96, 64*64),
            # ReLU(),
            Linear(cfg.IMAGE_SIZE*3, 64*64),
            ReLU(),
			Linear(64*64, 32*32),
            ReLU(),
			Linear(32*32, 16*16),
            ReLU(),
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 784
		self.decoder = Sequential(
			Linear(16*16, 32*32),
			ReLU(),
            Linear(32*32, cfg.SAMPLE_SIZE*3),
			Tanh()
		)
    

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class CVAE(Module):	

	def __init__(self):
		super(CVAE, self).__init__()

		kernel_size = 4 # (4, 4) kernel
		init_channels = 8 # initial number of filters
		image_channels = 3 # RGB images
		latent_dim = 16 # latent dimension for sampling

		# encoder
		self.encoder = Sequential(
			Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1),
			ReLU(),
			Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, stride=2, padding=1),
			ReLU(),
			Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, stride=2, padding=1),
			ReLU(),
			Conv2d(in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, stride=2, padding=0),
			ReLU()
		)

		# fully connected layers for learning representations
		self.fc1 = Linear(64, 128)
		self.fc_mu = Linear(128, latent_dim)
		self.fc_log_var = Linear(128, latent_dim)
		self.fc2 = Linear(latent_dim, 64)

		# decoder
		self.decoder = Sequential( 
			ConvTranspose2d(in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, stride=1, padding=0),
			ReLU(),
			ConvTranspose2d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, stride=2, padding=1),
			ReLU(),
			ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, stride=2, padding=1),
			ReLU(),
			ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, stride=2, padding=1),
			ReLU(),
			Flatten(),
			Linear(3*32*32, cfg.SAMPLE_SIZE*3),
			Tanh()
		)

	def reparameterize(self, mu, log_var):
		"""
		:param mu: mean from the encoder's latent space
		:param log_var: log variance from the encoder's latent space
		"""
		std = torch.exp(0.5*log_var) # standard deviation
		eps = torch.randn_like(std) # `randn_like` as we need the same size
		sample = mu + (eps * std) # sampling
		return sample

	def forward(self, x):
		# encoding
		x = self.encoder(x)
		batch, _, _, _ = x.shape
		x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
		hidden = self.fc1(x)
		# get `mu` and `log_var`
		mu = self.fc_mu(hidden)
		log_var = self.fc_log_var(hidden)
		# get the latent vector through reparameterization
		z = self.reparameterize(mu, log_var)
		z = self.fc2(z)
		z = z.view(-1, 64, 1, 1)

		# decoding
		reconstruction = self.decoder(z)		
		return reconstruction, mu, log_var

class CAE_new(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(CAE_new, self).__init__()

		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 4 * 1 * 1
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 256 * 8 * 8
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Tanh()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded

class CAE_AHZ(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(CAE_AHZ, self).__init__()

		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Tanh()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded

class PSGN_Vanilla(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(PSGN_Vanilla, self).__init__()

		self.c1 = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)
		
		self.c2 = Sequential(
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c3 = Sequential(
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c4 = Sequential(
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c5 = Sequential(
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c6 = Sequential(
			Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c7 = Sequential(
			Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2), # 8 * 4 * 4
			ReLU(),	
		)

		self.fc1 = Sequential(
			Linear(512*2*2, 2048),
			ReLU(),
			Linear(2048, 1024*3),
			ReLU(),
			Linear(1024*3, 1024*3),
			ReLU(),
		)
		
		self.d1 = Sequential(
			ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2) # 8 * 5 * 5	
		)
		
		self.c8 = Sequential(
			Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.d2 = Sequential(
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 5 * 5	
			ReLU(),
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=1) # 8 * 5 * 5	
		)

		self.c9 = Sequential(
			Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.d3 = Sequential(
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 5 * 5	
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1), # 8 * 5 * 5	
			ReLU()
		)

		self.c10 = Sequential(
			Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.c11 = Sequential(
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)


	def forward(self, x):

		x0 = self.c1(x)
		x1 = self.c2(x0)
		x2 = self.c3(x1)
		x3 = self.c4(x2)
		x4 = self.c5(x3)
		x5 = self.c6(x4)
		x = self.c7(x5)
		x = self.fc1(torch.flatten(x, 1))
		x = x.reshape(-1, 1024, 3)
		# print(x.shape)

		return x

class PSGN(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(PSGN, self).__init__()

		self.c1 = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)
		
		self.c2 = Sequential(
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c3 = Sequential(
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c4 = Sequential(
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c5 = Sequential(
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c6 = Sequential(
			Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c7 = Sequential(
			Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2), # 8 * 4 * 4
			ReLU(),	
		)

		self.fc1 = Sequential(
			Linear(512*2*2, 2048),
			ReLU(),
			Linear(2048, 1024),
			ReLU(),
			Linear(1024, 63*3),
			ReLU()
		)
		
		self.d1 = Sequential(
			ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2) # 8 * 5 * 5	
		)
		
		self.c8 = Sequential(
			Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.d2 = Sequential(
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 5 * 5	
			ReLU(),
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=1) # 8 * 5 * 5	
		)

		self.c9 = Sequential(
			Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.d3 = Sequential(
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 5 * 5	
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1), # 8 * 5 * 5	
			ReLU()
		)

		self.c10 = Sequential(
			Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.c11 = Sequential(
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)


	def forward(self, x):

		x0 = self.c1(x)
		x1 = self.c2(x0)
		x2 = self.c3(x1)
		x3 = self.c4(x2)
		x4 = self.c5(x3)
		x5 = self.c6(x4)
		x = self.c7(x5)
		x_additional = self.fc1(torch.flatten(x, 1))
		x_additional = x_additional.reshape(-1, 63, 3)
		x = self.d1(x)
		x5 = self.c8(x5)
		x = F.relu(torch.add(x, x5))
		x = self.d2(x)
		x4 = self.c9(x4)
		x = F.relu(torch.add(x, x4))
		x = self.d3(x)
		x3 = self.c10(x3)
		x = F.relu(torch.add(x, x3))
		x = self.c11(x)
		x = x.reshape(-1, 31*31, 3)
		x = torch.concat([x_additional, x], dim=1)
		# print(x.shape)

		return x

class Pixel2Point(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""

	def __init__(self):
		super(Pixel2Point, self).__init__()

		self.FEATURES_NUM = 256

		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=128, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
		)
		
		self.decoder = Sequential(
			Flatten(),
			Linear(self.FEATURES_NUM, self.FEATURES_NUM),
			Linear(self.FEATURES_NUM, self.FEATURES_NUM),
			Linear(self.FEATURES_NUM, self.FEATURES_NUM),
			ReLU(),
			Linear(self.FEATURES_NUM, cfg.SAMPLE_SIZE*3),
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class Pixel2Point_InitialPC(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""

	def __init__(self):
		super(Pixel2Point_InitialPC, self).__init__()

		self.FEATURES_NUM = 256
		self.INITIAL_SPHERE_POINTS = 16

		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=64, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
		)
		
		self.decoder = Sequential(
			Flatten(),
			Linear(self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM), self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM)),
			Linear(self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM), self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM)),
			Linear(self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM), self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM)),
			ReLU(),
			Linear(self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM), cfg.SAMPLE_SIZE*3),
		)

		self.initialSphere = torch.tensor([
										0.382683, 0.0, 0.92388,
										-0.382683, 0.0, 0.92388,
										0.92388, 0.0, 0.382683,
										0.46194, 0.800103, 0.382683,
										-0.46194, 0.800103, 0.382683,
										-0.92388, 0.0, 0.382683,
										-0.46194, -0.800103, 0.382683,
										0.46194, -0.800103, 0.382683,
										0.92388, 0.0, -0.382683,
										0.46194, 0.800103, -0.382683,
										-0.46194, 0.800103, -0.382683,
										-0.92388, 0.0, -0.382683,
										-0.46194, -0.800103, -0.382683,
										0.46194, -0.800103, -0.382683,
										0.382683, 0.0, -0.92388,
										-0.382683, 0.0, -0.92388

												]).reshape((self.INITIAL_SPHERE_POINTS, 3)).to(device="cuda")


	def forward(self, x):
		encoded = self.encoder(x)
		encoded = torch.flatten(encoded, 2).reshape(-1, self.FEATURES_NUM)
		encoded = torch.transpose(encoded.unsqueeze(2).expand(-1, -1, self.INITIAL_SPHERE_POINTS), 1, 2)
		
		sphere = self.initialSphere
		# sphere = sphere.unsqueeze(0).expand(1, -1, -1).to('cpu')
		sphere = sphere.unsqueeze(0).expand(cfg.BATCH_SIZE, -1, -1)
		# print(encoded.shape)
		# print(sphere.shape)
		encoded = torch.concat([sphere, encoded], dim=-1)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded

class CAE_AHZ_Attention(Module):
	def __init__(self):
		super(CAE_AHZ_Attention, self).__init__()
		# TODO: Initialize myModel
		
		self.sequence_length = 256
		self.input_size = 256
		self.embed_size = 128
		self.positional_encoding = torch.nn.Parameter(torch.rand(self.embed_size*2, self.embed_size*2))

		self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4
		self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 4 * 2 * 2
		self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 4 * 2 * 2
		self.conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 4 * 2 * 2
		self.conv5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # 4 * 2 * 2

		self.attention1 = MultiheadAttention(128, 32)

		self.deconv1 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1) # 8 * 5 * 5
		self.deconv2 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1) # 8 * 5 * 5
		self.deconv3 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1) # 4 * 15 * 15
		self.deconv4 = ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1) # 4 * 15 * 15
		
		self.linear1 = Linear(32*18*18, 2700)
		self.linear2 = Linear(2700, cfg.SAMPLE_SIZE*3)

		self.maxpool = MaxPool2d(kernel_size=2, stride=2)



	def forward(self, x):
		batch_size, channels, sequence_length, input_size = x.shape
        
		# Positional encoding
		x = x.reshape(batch_size*channels, sequence_length, -1)
		for i in range(batch_size*channels):
			x[i] = torch.add(x[i], self.positional_encoding)
		x = torch.unsqueeze(x, dim=0)
		x = x.reshape(batch_size, channels, sequence_length, input_size)
		batch_size, channels, sequence_length, input_size = x.shape


		# Conv 1
		x = torch.relu(self.conv1(x))
		x = self.maxpool(x)
		batch_size, channels, sequence_length, input_size = x.shape

		# Attn 1
		x = x.reshape(batch_size*channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention1(x, x, x)
		x = torch.relu(attn_output)

		# Conv 2		
		x = torch.unsqueeze(x, dim=0)
		x = x.reshape(batch_size, channels, sequence_length, input_size)
		x = torch.relu(self.conv2(x))
		x = self.maxpool(x)

		# Conv3
		x = torch.relu(self.conv3(x))
		x = self.maxpool(x)

		# Conv4
		x = torch.relu(self.conv4(x))
		x = self.maxpool(x)


		# # Conv 5
		# x = torch.relu(self.conv5(x))

		# Deconv 1-4
		# x = torch.relu(self.deconv1(x))
		x = torch.relu(self.deconv2(x))
		x = torch.relu(self.deconv3(x))
		# x = torch.relu(self.deconv4(x))


		# Linear 1-2
		batch_size, channels, height, width = x.shape
		x = x.reshape(-1, channels*height*width)
		x = torch.relu(self.linear1(x))
		x = self.linear2(x)
		x = torch.tanh(x)

		return x


class PureAttention(Module):
	
	def __init__(self):
		super(PureAttention, self).__init__()
		# TODO: Initialize myModel
		
		self.sequence_length = 256
		self.input_size = 256
		self.embed_size = 256		
		self.positional_encoding = torch.nn.Parameter(torch.rand(self.embed_size, self.embed_size))

		self.attention1 = MultiheadAttention(256, 32)
		self.attention2 = MultiheadAttention(128, 32)
		self.attention3 = MultiheadAttention(64, 16)
		self.attention4 = MultiheadAttention(32, 8)
		self.attention5 = MultiheadAttention(16, 4)
		
		self.linear1 = Linear(3*8*8, 3*12*12)
		self.linear2 = Linear(3*12*12, 768)
		self.linear3 = Linear(768, cfg.SAMPLE_SIZE*3)
		self.linear4 = Linear(cfg.SAMPLE_SIZE*3, cfg.SAMPLE_SIZE*3)

		self.deconv1 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1) # 8 * 5 * 5
		self.deconv2 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1) # 8 * 5 * 5
		self.deconv3 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1) # 4 * 15 * 15


		self.maxpool = MaxPool2d(kernel_size=2, stride=2)



	def forward(self, x):
		batch_size, channels, sequence_length, input_size = x.shape
        
		# Positional encoding
		x = x.reshape(batch_size*channels, sequence_length, -1)
		for i in range(batch_size*channels):
			x[i] = torch.add(x[i], self.positional_encoding)
		x = torch.unsqueeze(x, dim=0)
		x = x.reshape(batch_size, channels, sequence_length, input_size)

		# Attention layer 1
		batch_size, channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size*channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention1(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)

		# Attention layer 2
		batch_size_channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size_channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention2(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)

		# Attention layer 3
		batch_size_channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size_channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention3(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)

		# Attention layer 4
		batch_size_channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size_channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention4(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)


		# Attention layer 5
		batch_size_channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size_channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention5(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)



		# Deconv 1-2
		batch_size_channels, sequence_length, input_size = x.shape
		x = torch.unsqueeze(x, dim=0)
		x = x.reshape(int(batch_size_channels/3), 3, sequence_length, input_size)
		# x = self.deconv1(x)
		# x = self.deconv2(x)
		# x = self.deconv3(x)


		# Linear 1-2
		# batch_size_channels, sequence_length, input_size = x.shape
		# x = torch.unsqueeze(x, dim=0)
		# x = x.reshape(int(batch_size_channels/3), 3, sequence_length, input_size)
		batch_size, channels, height, width = x.shape
		x = x.reshape(-1, channels*height*width)
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.linear2(x))
		x = torch.relu(self.linear3(x))
		x = self.linear4(x)
		x = torch.tanh(x)

		return x


class PreLayerNormAttention(Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = Sequential(
            Linear(embed_dim, hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, embed_dim),
            Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = Sequential(*[PreLayerNormAttention(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = Sequential(
            LayerNorm(embed_dim),
            Linear(embed_dim, num_classes)
        )
        self.dropout = Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = torch.nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = torch.nn.Parameter(torch.randn(1,1+num_patches,embed_dim))


    def forward(self, x):
        # Preprocess input
        x = imageToPatches(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


