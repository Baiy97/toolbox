import torch
import numpy as np
import cv2


def CAM(image_data, feature_maps, weight_matrix, target_index=0):
	'''
	batch_data:  data
	feature_maps:  bsxCxHxW
	weight_matrix:	cxC
	target_index: label of data
	'''
	to_H, to_W = 480, 480		# cam out size
	bs, C, H, W = feature_maps.shape

	# cam make
	weight_softmax = torch.softmax(weight_matrix, dim=0).cpu().data.numpy()
	feature_maps = feature_maps.cpu().data.numpy()
	cam = weight_softmax[target_index].reshape(1, C, 1, 1) * feature_maps
	cam = np.sum(cam, axis=1)
	min_cam = np.min(cam.reshape(-1, H * W), axis=-1).reshape(-1, 1, 1)
	max_cam = np.max(cam.reshape(-1, H * W), axis=-1).reshape(-1, 1, 1)
	cam = (cam - min_cam) / max_cam

	# show
	image = (image_data * 0.5 + 0.5).cpu().data.numpy()			# unnormalize if need
	for ind in range(bs):
		img = image[ind].transpose(1, 2, 0)
		mask = cv2.resize(cam[ind], (to_W, to_H))
		heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
		heatmap = np.float32(heatmap) / 255
		out = heatmap * 0.3 + np.float32(img) * 0.7			# adjust weight
		out = out / np.max(out)
		out = np.uint8(255 * out)
		cv2.imwrite('cam_{}'.format(ind), out)