# Python version = 3.6.8
# PyTorch version = 1.0.1
# Ubuntu 18.04

import torch
import torch.nn as nn
import math
from thop import profile

# 3x3 convolution with padding
def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
		padding=1, bias=False)

class CBAM_channel(nn.Module):
	def __init__(self, out_channels, reduction=16, dilation=4):
		super().__init__()
		# 2D adaptive average pooling
		self.average_pool = nn.AdaptiveAvgPool2d(1)
		# 2D adaptive max pooling
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		self.fc1  = nn.Linear(out_channels, out_channels//reduction, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2  = nn.Linear(out_channels//reduction, out_channels, bias=False)
		self.sigmoid   = nn.Sigmoid()

	def forward(self, x):
		out = x

		out_c1 = self.average_pool(out)
		out_c1 = out_c1.view(out_c1.size(0), -1) # NxC
		out_c1 = self.fc1(out_c1)
		out_c1 = self.relu(out_c1)
		out_c1 = self.fc2(out_c1)

		out_c2 = self.max_pool(out)
		out_c2 = out_c2.view(out_c2.size(0), -1) # NxC
		out_c2 = self.fc1(out_c2)
		out_c2 = self.relu(out_c2)
		out_c2 = self.fc2(out_c2)

		out = out_c1 + out_c2
		out = self.sigmoid(out)
		out = out.view(out.size(0), out.size(1), 1, 1) # NxCx1x1
		out = out.expand_as(x) # NxCxHxW

		out = x * out
		return out

class CBAM_spatial(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
		self.bn1   = nn.BatchNorm2d(1)
		self.relu  = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		out = x
		# max pooling along channel axis
		out_s1 = torch.max(out, 1)[0].unsqueeze(1) #Nx1xHxW
		# mean pooling along channel axis
		out_s2 = torch.mean(out, 1).unsqueeze(1) #Nx1xHxW
		# concatenate out_s1 and out_s2 along channel axis
		out = torch.cat((out_s1, out_s2), 1) #Nx2xHxW
		out = self.conv1(out)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.sigmoid(out) # Nx1xHxW
		out = out.expand_as(x) # NxCxHxW
		out = x * out
		return out

class Res_CBAM_BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, channel=0, spatial=0, stride=1,
		downsample=None, reduction=16, dilation=4):
		super().__init__()
		self.conv1 = conv3x3(in_channels, out_channels, stride)
		self.bn1   = nn.BatchNorm2d(out_channels)
		self.relu  = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(out_channels, out_channels)
		self.bn2   = nn.BatchNorm2d(out_channels)
		# downsample for the residual
		self.downsample = downsample
		self.stride = stride

		#### bottleneck attention module ####
		# channel attention #
		self.channel = channel
		if self.channel == 1:
			self.cbam_channel = CBAM_channel(out_channels, reduction, dilation)
		# channel attention #
		self.spatial = spatial
		if self.spatial == 1:
			self.cbam_spatial = CBAM_spatial()
		#### bottleneck attention module ####

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		#### bottleneck attention module ####
		out_temp = out
		# channel attention #
		if (self.channel is 1) and (self.spatial is 0):
			out = self.cbam_channel(out)
			#print ("resnet34_cbam_c")
		# spatial attention #
		elif (self.channel is 0) and (self.spatial is 1):
			out = self.cbam_spatial(out)
			#print ("resnet34_cbam_s")
		# combination
		elif (self.channel is 1) and (self.spatial is 1):
			out_c = self.cbam_channel(out)
			out_s = self.cbam_spatial(out_c)
			out = out_s
			#print ("resnet34_cbam")
		# warning
		else:
			print ('This is not ResNet + CBAM. Please check the model.')
		#### bottleneck attention module ####

		# to make sizes of the residual and the output be same
		if self.downsample is not None:
			residual = self.downsample(x)
		out = out + residual
		out = self.relu(out)
		return out

class Res_CBAM_Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels, channel=0, spatial=0, stride=1,
		downsample=None, reduction=16, dilation=4):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn1   = nn.BatchNorm2d(out_channels)
		self.relu  = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(out_channels, out_channels, stride)
		self.bn2   = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, bias=False)
		self.bn3   = nn.BatchNorm2d(out_channels*4)
		self.downsample = downsample
		self.stride = stride

		#### bottleneck attention module ####
		# channel attention #
		self.channel = channel
		if self.channel == 1:
			self.cbam_channel = CBAM_channel(out_channels*4, reduction, dilation)
		# spatial attention #
		self.spatial = spatial
		if self.spatial == 1:
			self.cbam_spatial = CBAM_spatial()
		#### bottleneck attention module ####

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)

		#### bottleneck attention module ####
		out_temp = out
		# channel attention #
		if (self.channel is 1) and (self.spatial is 0):
			out = self.cbam_channel(out)
			#print ("resnet50_cbam_c")
		# spatial attention #
		elif (self.channel is 0) and (self.spatial is 1):
			out = self.cbam_spatial(out)
			#print ("resnet50_cbam_s")
		# combination
		elif (self.channel is 1) and (self.spatial is 1):
			out_c = self.cbam_channel(out)
			out_s = self.cbam_spatial(out_c)
			out = out_s
			#print ("resnet50_cbam")
		# warning
		else:
			print ('This is not ResNet + CBAM. Please check the model.')
		#### bottleneck attention module ####

		# to make sizes of the residual and the output be same
		if self.downsample is not None:
			residual = self.downsample(x)
		out = out + residual
		out = self.relu(out)
		return out

class ResNet_CBAM(nn.Module):
	def __init__(self, block, num_layers, flag_c, flag_s, num_classes=100):
		super().__init__()
		self.res_in_channels = 64
		self.conv1 = nn.Conv2d(3, self.res_in_channels, kernel_size=3, stride=1,
			padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.res_in_channels)
		self.relu = nn.ReLU(inplace=True)
		self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.layer1 = self._make_layer(block, 64, num_layers[0], flag_c, flag_s)
		self.layer2 = self._make_layer(block, 128, num_layers[1], flag_c, flag_s, stride=2)
		self.layer3 = self._make_layer(block, 256, num_layers[2], flag_c, flag_s, stride=2)
		self.layer4 = self._make_layer(block, 512, num_layers[3], flag_c, flag_s, stride=2)
		self.average_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear((512*block.expansion), num_classes)

		# parameter initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2.0/n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, out_channels, num_blocks, flag_c, flag_s, stride=1):
		'''
		Args:
			block: which block to be used to create ResNet
			out_channels: number of the channels of output
			              BasicBlock: out_channels
		              Bottleneck: out_channels*4
			num_blocks: number of the blocks in one layer
		Returns:
			nn.sequential(*layers)
		'''

		downsample = None
		# to make sizes of the residual and the output be same
		# used in the first block in each layer (excepet layer1 in resnet34)
		if (stride!=1) or (self.res_in_channels!=out_channels*block.expansion):
			downsample = nn.Sequential(
				nn.Conv2d(self.res_in_channels, out_channels*block.expansion,
					kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels*block.expansion))

		layers = []
		layers.append(block(self.res_in_channels, out_channels, flag_c, flag_s,
			stride, downsample))
		self.res_in_channels = out_channels * block.expansion
		for i in range(1, num_blocks):
			layers.append(block(self.res_in_channels, out_channels, flag_c, flag_s))
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.max_pool(out)

		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)

		out = self.average_pool(out)
		out = self.fc(out.view(out.size(0), -1))
		return out

def resnet34_cbam_c(pretrained=False, **kwargs):
	model = ResNet_CBAM(Res_CBAM_BasicBlock, [3,4,6,3], 1, 0, **kwargs)
	flops, params = profile(model, input_size=(1, 3, 32,32),
		custom_ops={ResNet_CBAM: model})
	print ("flops & params:", flops, params)
	return model

def resnet34_cbam_s(pretrained=False, **kwargs):
	model = ResNet_CBAM(Res_CBAM_BasicBlock, [3,4,6,3], 0, 1, **kwargs)
	flops, params = profile(model, input_size=(1, 3, 32,32),
		custom_ops={ResNet_CBAM: model})
	print ("flops & params:", flops, params)
	return model

def resnet34_cbam(pretrained=False, **kwargs):
	model = ResNet_CBAM(Res_CBAM_BasicBlock, [3,4,6,3], 1, 1, **kwargs)
	flops, params = profile(model, input_size=(1, 3, 32,32),
		custom_ops={ResNet_CBAM: model})
	print ("flops & params:", flops, params)
	return model

def resnet50_cbam_c(pretrained=False, **kwargs):
	model = ResNet_CBAM(Res_CBAM_Bottleneck, [3,4,6,3], 1, 0, **kwargs)
	flops, params = profile(model, input_size=(1, 3, 32,32),
		custom_ops={ResNet_CBAM: model})
	print ("flops & params:", flops, params)
	return model

def resnet50_cbam_s(pretrained=False, **kwargs):
	model = ResNet_CBAM(Res_CBAM_Bottleneck, [3,4,6,3], 0, 1, **kwargs)
	flops, params = profile(model, input_size=(1, 3, 32,32),
		custom_ops={ResNet_CBAM: model})
	print ("flops & params:", flops, params)
	return model

def resnet50_cbam(pretrained=False, **kwargs):
	model = ResNet_CBAM(Res_CBAM_Bottleneck, [3,4,6,3], 1, 1, **kwargs)
	flops, params = profile(model, input_size=(1, 3, 32,32),
		custom_ops={ResNet_CBAM: model})
	print ("flops & params:", flops, params)
	return model