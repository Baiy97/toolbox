import os
import torch
from tensorboardX import SummaryWriter


class Logger():
	def __init__(self, log_path='log'):
		self.logger = SummaryWriter(log_path)
		self.add_images_maxnum = 10

	def add_scalar(self, tag, value, step):
		if isinstance(value, dict):
			for k, v in value.items():
				self.logger.add_scalar(k, v.mean().item(), step)
		else:
			self.logger.add_scalar(tag, value, step)

	def add_image(self, tag, value, step, dataformats='CHW'):
		self.logger.add_image(tag, value, step)

	def add_images(self, tag, value, step, dataformats='CHW'):
		value = value[:self.add_images_maxnum]
		self.logger.add_images(tag, value, step)

	def add_graph(self, model, input_to_model=None):
		if input_to_model is None:
			input_to_model = torch.zeros((256, 3, 32, 128))
		self.add_graph(model, input_to_model)
