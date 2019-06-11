import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class ImageFolder(data.Dataset):
	def __init__(self,dataset,transform):
		"""Initializes image paths and preprocessing module."""
		self.dataset = dataset
		self.transform = transform
		
		
	def __getitem__(self,idx):
		image = self.dataset[idx]
		if self.transform is not None:
			image = self.transform(image)
		image.view(1,image.shape[-3],image.shape[-2],image.shape[-1])
		
		#s.append(image)
		
		return image

	def __len__(self):
		return len(self.dataset)

def get_loader(dataset):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(dataset)
	data_loader = data.DataLoader(dataset=dataset)
	return data_loader
