# Python version = 3.6.8
# PyTorch version = 1.0.1
# Reference:
# https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html#cifar-100-dataset
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from tensorboardX import SummaryWriter
import numpy as np

def data_loader(data_root, batch_size_train, batch_size_test):
	'''
	Args:
		data_root: root directory of the data
		batch_size_train: mini-batch size of training set
		batch_size_test: mini-batch size of test size
	Returns:
		train_loader: laoder for training set
		test_loader: loader for test set
	'''

	# normalize training set together with augmentation
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.507,0.487,0.441), (0.267,0.256,0.276))])
	# normalize test set as traing set without augmentation
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.507,0.487,0.441), (0.267,0.256,0.276))])
	# load cifar-100
	print ("=====> prepaing CIFAR-100...")
	trainset = torchvision.datasets.CIFAR100(root=data_root, train=True,
		download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
		shuffle=True, num_workers=4)

	testset = torchvision.datasets.CIFAR100(root=data_root, train=False,
		download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size_test,
		shuffle=True, num_workers=2)
	return trainloader, testloader

def calculate_accuracy(net, loader, is_gpu):
	'''
	Args:
		net: network model used
		loader (torch.utils.data.Dataloader): training / test set loader
		is_gpu (bool): whether run on GPU
	Returns:
		accuracy: overall accuracy
	'''

	correct = 0.0
	total = 0.0
	for data in loader:
		images, labels = data
		if is_gpu:
			images = images.cuda()
			labels = labels.cuda()
		outputs = net(Variable(images))
		_, predicted = torch.max(outputs.data, 1)

		#print (labels.size()) -> out:torch.Size([256])
		total = total + labels.size(0)
		correct = correct + (predicted == labels).sum()
	accuracy = 100 * correct / total
	return accuracy

class AverageMeter(object):
	# Update average when receiving new data
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, n=1):
		self.val = val
		self.sum = self.sum + val*n
		self.count = self.count + n
		self.avg = self.sum / self.count

def topk_error(output, label, topk=(1,)):
	'''
	Args:
		output: output the model used
		label: real label of the image
	Returns:
		res: top k accuracy
	'''
	maxk = max(topk)
	batch_size = output.size(0)

	# get k largest numbers
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(label.view(1,-1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res

def train(net, criterion, optimizer, trainloader, testloader, start_epoch, epochs, is_gpu,
	save_ckpt, log_root):
	'''
	Args:
		net: network model to be trained
		criterion: CrossEntropyLoss
		optimizer: SGD with momentum optimizer
		trainloader: training set loader
		testloader: test set loader
		start_spoch: checkpoint saved epoch
		epochs: training epochs
		is_gpu: whether run on GPU
		log_root: for visualization
	'''

	print ("=====> start training...")

	#writer = SummaryWriter(log_dir=log_root)
	#data_train_accu = np.zeros(shape=(epochs,1))
	#data_test_accu = np.zeros(shape=(epochs,1))
	#data_running_loss = np.zeros(shape=(epochs,1))
	data_train_t1 = np.zeros(shape=(epochs,1))
	data_train_t5 = np.zeros(shape=(epochs,1))
	data_test_t1 = np.zeros(shape=(epochs,1))
	data_test_t5 = np.zeros(shape=(epochs,1))
	data_loss = np.zeros(shape=(epochs,1))
	for epoch in range(start_epoch, epochs+start_epoch):
		# switch to training mode
		net.train()
		#running_loss = 0.0
		top1_train = AverageMeter()
		top5_train = AverageMeter()
		top1_test = AverageMeter()
		top5_test = AverageMeter()
		loss_ = AverageMeter()
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			if is_gpu:
				inputs = inputs.cuda()
				labels = labels.cuda()
			inputs = Variable(inputs)
			labels = Variable(labels)

			# compute output and loss
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			
			# compute gardient and do optimization step
			optimizer.zero_grad()			
			loss.backward()
			optimizer.step()
			
			pred1_train, pred5_train = topk_error(outputs.data, labels.data, topk=(1,5))
			top1_train.update(100.0-pred1_train.item(), inputs.size(0))
			top5_train.update(100.0-pred5_train.item(), inputs.size(0))
			#print ("loss.item(): ", loss.item())
			#print ("loss.data[0]:", loss.data[0])
			#running_loss = running_loss + loss.item()
			loss_.update(loss.item(), inputs.size(0))
		#running_loss = running_loss / len(trainloader)
		#train_accuracy = calculate_accuracy(net, trainloader, is_gpu)
		print ("Iteration: {0}".format(epoch+1))
		print ("Train | Top 1: {0}% | Top 5: {1}% | Loss: {2}"
			.format(top1_train.avg, top5_train.avg, loss_.avg))

		# switch to test mode
		net.eval()
		for i, data in enumerate(testloader, 0):
			inputs, labels = data
			if is_gpu:
				inputs = inputs.cuda()
				labels = labels.cuda()
			inputs = Variable(inputs)
			labels = Variable(labels)
			outputs = net(inputs)

			pred1_test, pred5_test = topk_error(outputs.data, labels.data, topk=(1,5))
			top1_test.update(100.0-pred1_test.item(), inputs.size(0))
			top5_test.update(100.0-pred5_test.item(), inputs.size(0))

		#test_accuracy = calculate_accuracy(net, testloader, is_gpu)
		print ("Test  | Top 1: {0}% | Top 5: {1}%".format(top1_test.avg, top5_test.avg))

		# show the loss and accuracy on Tensorboard
		#writer.add_scalar('Loss', running_loss, epoch+1)
		#writer.add_scalar('Train/Accu', train_accuracy, epoch+1)
		#writer.add_scalar('Test/Accu', test_accuracy, epoch+1)
		#data_train_accu[epoch] = train_accuracy.cpu()
		#data_test_accu[epoch] = test_accuracy.cpu()
		#data_running_loss[epoch] = running_loss

		# record data 
		data_train_t1[epoch] = top1_train.avg
		data_train_t5[epoch] = top5_train.avg
		data_test_t1[epoch] = top1_test.avg
		data_test_t5[epoch] = top5_test.avg
		data_loss[epoch] = loss_.avg

		# save model
		if ((epoch==0) or (epoch==epochs-1)):
			print ("=====> saving model...")
			#state = {'net': net.module if is_gpu else net, 'epoch': epoch}
			#state = {'net': net.state_dict(), 'epoch': epoch}
			if not os.path.isdir(save_ckpt):
				os.makedirs(save_ckpt)
			#torch.save(state, save_ckpt+'/ckpt.t7')
			#torch.save(state, save_ckpt+'/ckpt.pth')
			#torch.save(net, save_ckpt+'/ckpt.pth')
			torch.save({
				'epoch': epoch+1,
				'state_dict': net.module.state_dict() if is_gpu else net.state_dict()},
				save_ckpt+'/ckpt.pth')

	# save files recording data
	if not os.path.isdir(log_root):
		os.makedirs(log_root)
	np.savetxt(log_root+'running_loss', data_running_loss)
	np.savetxt(log_root+'/train_top1', data_train_t1)
	np.savetxt(log_root+'/train_top5', data_train_t5)
	np.savetxt(log_root+'/test_top1', data_test_t1)
	np.savetxt(log_root+'/test_top5', data_test_t5)
	np.savetxt(log_root+'/loss', data_loss)
	print ("=====> finish training...")