from torch.nn import Sequential
from torch.nn import Module         # Rather than using the Sequential PyTorch class to implement LeNet, we’ll instead subclass the Module object so you can see how PyTorch implements neural networks using classes
from torch.nn import Conv2d         # PyTorch’s implementation of convolutional layers
from torch.nn import Linear         # Fully connected layers
from torch.nn import MaxPool2d      # Applies 2D max-pooling to reduce the spatial dimensions of the input volume
from torch.nn import ReLU           # ReLU activation function
from torch.nn import Tanh
import sys
import numpy as np
from torch.nn.modules.activation import Tanh
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.flatten import Flatten
sys.path.append('/home/ahz/Desktop/3D-Reconstruction/3D-Reconstruction')
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
