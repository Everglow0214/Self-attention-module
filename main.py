# Python version = 3.6.8
# PyTorch version = 1.0.1
# Ubuntu 18.04

# Reference:
# https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html#cifar-100-dataset
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

# pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from cifar_100 import *
from resnet_base import resnet34, resnet50
from resnet_se import resnet34_se, resnet50_se
from resnet_bam import resnet34_bam_c, resnet34_bam_s, resnet34_bam
from resnet_bam import resnet50_bam_c, resnet50_bam_s, resnet50_bam
from resnet_cbam import resnet34_cbam_c, resnet34_cbam_s, resnet34_cbam
from resnet_cbam import resnet50_cbam_c, resnet50_cbam_s, resnet50_cbam

#from torchvision.models.resnet import resnet34, resnet50
#from thop import profile

import argparse
import os

# choose GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()

# directory
parser.add_argument('--data_root', type=str,
	default="../data", help='path for data set')
parser.add_argument('--ckpt_root', type=str,
	default="/checkpoint/se/ckpt.pth", help='path for reloading the model')
parser.add_argument('--save_ckpt', type=str,
	default="checkpoint/res50_cbam", help='path for saving the model')
parser.add_argument('--log_root', type=str,
	default="logs/res50_cbam", help='path for the logs (loss and accuracy)')
# hyperparameter settings
parser.add_argument('--lr', type=float,
	default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float,
	default=0.9, help='momentum factor')
parser.add_argument('--weight_decay',type=float,
	default=1e-5, help='weight decay (L2 penalty)')
parser.add_argument('--epochs', type=int,
	default=100, help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int,
	default=64, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int,
	default=64, help='test set input batch size')
# training settings
parser.add_argument('--resume', type=bool,
	default=False, help='whether re-training from ckpt')
parser.add_argument('--is_gpu', type=bool,
	default=True, help='whether training with GPU')

# parse the arguments
args = parser.parse_args()

def main():
	start_epoch = 0

	if args.resume:
		# resume training from the last time
		# load checkpoint
		print ("=====> resuming from checkpoint...")
		assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found'
		checkpoint = torch.load(args.ckpt_root)
		net = checkpoint['net']
		start_epoch = checkpoint['epoch']
	else:
		print ("=====> building a new ResNet model...")
		# resnet34(), resnet50()
		# resnet34_se(), resnet50_se()
		# resnet34_bam_c(), resnet34_bam_s(), resnet34_bam()
		# resnet50_bam_c(), resnet50_bam_s(), resnet50_bam()
		# resnet34_cbam_c(), resnet34_cbam_s(), resnet34_cbam()
		# resnet50_cbam_c(), resnet50_cbam_s(), resnet50_cbam()
		net = resnet50_cbam()

	print ("=====> initializing CUDA support for ResNet model...")
	if args.is_gpu:
		net = torch.nn.DataParallel(net).cuda()
		cudnn.benchmark = True

	# loss function, optimizer and scheduler
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
		weight_decay=args.weight_decay)

	# load CIFAR-100
	trainloader, testloader = data_loader(args.data_root, args.batch_size_train,
		args.batch_size_test)

	# training
	train(net, criterion, optimizer, trainloader, testloader, start_epoch, args.epochs,
		args.is_gpu, args.save_ckpt, args.log_root)
	
	# get the number of model parameters
	num_param = 0
	for p in net.parameters():
		num_param = p.data.nelement() + num_param
	print ("=====> number of parameters:", num_param)
	#flops, params = profile(net, input_size=(1,3,32,32))
	#print ("=====> flops & params:", flops, params)

if __name__ == '__main__':
	main()
