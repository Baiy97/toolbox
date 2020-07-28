import sys
import numpy as np
import torch
import cv2

# TODO need feature & classifier paradigm
def show_cam_on_image(image, mask, filename):
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(image.transpose(1,2,0))			# can add weight
	cam = cam / np.max(cam)
	cam = np.uint8(255 * cam)
	cv2.imwrite('res/'+filename, cam)


class FeatureExtractor():
	'''
	return outputs of target layers and the feature output before fc
	'''
	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradients(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []
		for name, module in self.model._modules.items():
			if name == 'fc':
				break
			x = module(x)
			if name in self.target_layers:
				x.register_hook(self.save_gradients)
				outputs += [x]
		return outputs, x


class ModelOutputs():
	'''
	return activations and the final output
	'''
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		output = self.model.fc(output)
		return target_activations, output


class GradCam():
	def __init__(self,  model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		self.to_size = (480, 480)
		if use_cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(model, target_layer_names)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, targ_index=None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		# make something to backward
		if targ_index is None:	# if label is not specified
			targ_index = np.argmax(output.cpu().data.numpy())
		one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
		one_hot[0][targ_index] = 1
		one_hot = torch.from_numpy(one_hot).requires_grad_()
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		self.model.zero_grad()
		one_hot.backward()

		# make cam of the last target layer
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]
		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		weights = np.mean(grads_val, axis=(2, 3))[0, :]
		cam = np.zeros(target.shape[1:], dtype=np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]
		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, self.to_size)
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam


dataset, dataloader = None, None
model = None
model.eval()
cnt = 0
for i, data_i in enumerate(dataloader):
	input, _ = data_i		# type [image, label]
	for j in range(input.shape[0]):
		input_j = input[j].unsqueeze(0)
		image_j = (input[j] * 0.5 + 0.5).cpu().data.numpy()
		target_layer_names = ['layer4']
		grad_cam = GradCam(model, target_layer_names, use_cuda=True)
		mask = grad_cam(input_j, targ_index=0)
		file_name = dataset.image_list[cnt][0].split('/')[-1]
		show_cam_on_image(image_j, mask, file_name)
		cnt += 1
		if cnt % 100 == 0:
			print('{}/{}'.format(cnt, len(dataset)))
