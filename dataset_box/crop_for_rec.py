import os
import cv2
from PIL import Image

# deal with ic03
import xml.dom.minidom

def crop_ic03(is_train):
	if is_train:
		data_dir = '/Users/mtdp/Documents/datasets/ic03/SceneTrialTrain/'
		gt_file = '/Users/mtdp/Documents/datasets/ic03/SceneTrialTrain/segmentation.xml'
		save_dir = 'ic03_train'
	else:
		data_dir = '/Users/mtdp/Documents/datasets/ic03/SceneTrialTest/'
		gt_file = '/Users/mtdp/Documents/datasets/ic03/SceneTrialTest/segmentation.xml'
		save_dir = 'ic03_test'

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	dom = xml.dom.minidom.parse(gt_file)
	root = dom.documentElement
	total_cnt = 0
	# train_val split
	image_items = root.getElementsByTagName('image')
	for image_item in image_items:
		image_name = image_item.getElementsByTagName('imageName')[0].childNodes[0].data
		image = Image.open(os.path.join(data_dir, image_name))

		instances = image_item.getElementsByTagName('taggedRectangle')
		for cnt, ins in enumerate(instances):
			x = int(eval(ins.attributes.get('x').nodeValue))
			y = int(eval(ins.attributes.get('y').nodeValue))
			width = int(eval(ins.attributes.get('width').nodeValue))
			height = int(eval(ins.attributes.get('height').nodeValue))
			x = max(0, x-width*0.2)
			y = max(0, y-height*0.2)
			crop_image = image.crop((x, y, x+width*1.4, y+height*1.4))
			crop_image.save('{}/{}_{}.jpg'.format(save_dir, image_name.replace('/', '_'), cnt))
			total_cnt += 1
	if is_train:
		print('Train ic03: ', total_cnt)
	else:
		print('Test ic03: ', total_cnt)

def crop_ic13(is_train):
	if is_train:
		data_dir = '/Users/mtdp/Documents/datasets/ic13/Challenge2_Training_Task12_Images'
		gt_dir = '/Users/mtdp/Documents/datasets/ic13/Challenge2_Training_Task1_GT'
		save_dir = 'ic13_train'
	else:
		data_dir = '/Users/mtdp/Documents/datasets/ic13/Challenge2_Test_Task12_Images'
		gt_dir = '/Users/mtdp/Documents/datasets/ic13/Challenge2_Test_Task1_GT'
		save_dir = 'ic13_test'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	total_cnt = 0
	for image_file in os.listdir(data_dir):
		image = Image.open(os.path.join(data_dir, image_file))
		gt_file = os.path.join(gt_dir, 'gt_' + image_file.replace('.jpg', '.txt'))
		with open(gt_file, 'r') as f:
			lines = f.readlines()
		for cnt, line in enumerate(lines):
			if is_train:
				x1, y1, x2, y2 = [int(t) for t in line.strip().split()[:4]]
			else:
				x1, y1, x2, y2 = [int(t) for t in line.strip().split(', ')[:4]]
			width = x2 - x1
			height = y2 - y1
			x, y = x1, y1
			x = max(0, x-width*0.2)
			y = max(0, y-height*0.2)
			crop_image = image.crop((x, y, x+width*1.4, y+height*1.4))
			# crop_image = image.crop((x1, y1, x2, y2))
			crop_image.save('{}/{}_{}.jpg'.format(save_dir, image_file, cnt))
			total_cnt += 1
	if is_train:
		print('Train ic13: ', total_cnt)
	else:
		print('Test ic13: ', total_cnt)


def crop_ic15(is_train):
	if is_train:
		data_dir = '/Users/mtdp/Documents/datasets/ic15/train_images'
		gt_dir = '/Users/mtdp/Documents/datasets/ic15/train_gts'
		save_dir = 'ic15_train'
	else:
		data_dir = '/Users/mtdp/Documents/datasets/ic15/test_images'
		gt_dir = '/Users/mtdp/Documents/datasets/ic15/test_gts'
		save_dir = 'ic15_test'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	total_cnt = 0
	for image_file in os.listdir(data_dir):
		image = Image.open(os.path.join(data_dir, image_file))
		gt_file = os.path.join(gt_dir, 'gt_'+image_file.replace('.jpg', '.txt'))
		with open(gt_file, 'r') as f:
			lines = f.readlines()
		for cnt, line in enumerate(lines):
			line = line.strip().split(',', 8)
			if line[-1] == '###':
				continue
			line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
			loc = [int(t) for t in line[:-1]]
			x1, x2 = min(loc[::2]), max(loc[::2])
			y1, y2 = min(loc[1::2]), max(loc[1::2])
			width = x2 - x1
			height = y2 - y1
			x, y = x1, y1
			x = max(0, x-width*0.2)
			y = max(0, y-height*0.2)
			crop_image = image.crop((x, y, x+width*1.4, y+height*1.4))
			# crop_image = image.crop((x1, y1, x2, y2))
			crop_image.save('{}/{}_{}.jpg'.format(save_dir, image_file, cnt))
			total_cnt += 1
	if is_train:
		print('Train ic15: ', total_cnt)
	else:
		print('Test ic15: ', total_cnt)


def crop_totaltext(is_train):
	if is_train:
		data_dir = '/Users/mtdp/Documents/datasets/TotalText/Images/Train'
		gt_dir = '/Users/mtdp/Documents/datasets/TotalText/train_gts'
		save_dir = 'totaltext_train'
	else:
		data_dir = '/Users/mtdp/Documents/datasets/TotalText/Images/Test'
		gt_dir = '/Users/mtdp/Documents/datasets/TotalText/test_gts'
		save_dir = 'totaltext_test'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	total_cnt = 0
	for image_file in os.listdir(data_dir):
		image = Image.open(os.path.join(data_dir, image_file))
		gt_file = os.path.join(gt_dir, image_file+'.txt')
		with open(gt_file, 'r') as f:
			lines = f.readlines()
		for cnt, line in enumerate(lines):
			line = line.strip().split(',')
			if line[-1] == '###':
				continue
			try:
				loc = [int(t) for t in line[:-1]]
			except:
				loc = [int(t) for t in line[:-2]]
			x1, x2 = min(loc[::2]), max(loc[::2])
			y1, y2 = min(loc[1::2]), max(loc[1::2])
			width = x2 - x1
			height = y2 - y1
			x, y = x1, y1
			x = max(0, x-width*0.2)
			y = max(0, y-height*0.2)
			crop_image = image.crop((x, y, x+width*1.4, y+height*1.4))
			# crop_image = image.crop((x1, y1, x2, y2))
			crop_image.save('{}/{}_{}.jpg'.format(save_dir, image_file, cnt))
			total_cnt += 1
	if is_train:
		print('Train totaltext: ', total_cnt)
	else:
		print('Test totaltext: ', total_cnt)


def crop_TD500(is_train):
	if is_train:
		data_dir = '/Users/mtdp/Documents/datasets/TD_TR/TD500/train_images'
		gt_dir = '/Users/mtdp/Documents/datasets/TD_TR/TD500/train_gts'
		save_dir = 'TD500_train'
	else:
		data_dir = '/Users/mtdp/Documents/datasets/TD_TR/TD500/test_images'
		gt_dir = '/Users/mtdp/Documents/datasets/TD_TR/TD500/test_gts'
		save_dir = 'TD500_test'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	total_cnt = 0
	for image_file in os.listdir(data_dir):
		image = Image.open(os.path.join(data_dir, image_file))
		gt_file = os.path.join(gt_dir, image_file+'.txt')
		with open(gt_file, 'r') as f:
			lines = f.readlines()
		for cnt, line in enumerate(lines):
			line = line.strip().split(',', 8)
			if line[-1] == '1':
				continue
			loc = [int(t) for t in line[:-1]]
			x1, x2 = min(loc[::2]), max(loc[::2])
			y1, y2 = min(loc[1::2]), max(loc[1::2])
			width = x2 - x1
			height = y2 - y1
			x, y = x1, y1
			x = max(0, x-width*0.2)
			y = max(0, y-height*0.2)
			crop_image = image.crop((x, y, x+width*1.4, y+height*1.4))
			# crop_image = image.crop((x1, y1, x2, y2))
			crop_image.save('{}/{}_{}.jpg'.format(save_dir, image_file, cnt))
			total_cnt += 1
	if is_train:
		print('Train TD500: ', total_cnt)
	else:
		print('Test TD500: ', total_cnt)


if __name__ == '__main__':
	crop_ic03(is_train=True)
	crop_ic03(is_train=False)
	crop_ic13(is_train=True)
	crop_ic13(is_train=False)
	crop_ic15(is_train=True)
	crop_ic15(is_train=False)
	crop_totaltext(is_train=True)
	crop_totaltext(is_train=False)
	crop_TD500(is_train=True)
	crop_TD500(is_train=False)













