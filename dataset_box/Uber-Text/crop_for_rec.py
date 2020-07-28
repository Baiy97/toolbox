import os
from PIL import Image
from tqdm import tqdm

to_save_dir = '/workdir/datasets/Uber-Text-Rec'
data_dir = '/workdir/datasets/Uber-Text'
stages = [
	'val/1Kx1K',
	'val/4Kx4K',
	'test/1Kx1K',
	'test/4Kx4K',
	'train/1Kx1K',
	'train/4Kx4K',]

for stage in stages:
	label_files = [t for t in os.listdir(os.path.join(data_dir, stage)) if t.endswith('.txt')]
	rec_gt_list = []
	os.makedirs(os.path.join(to_save_dir, stage))
	print(stage)
	for label_file in tqdm(label_files):
		image_name = label_file.replace('truth_', '').replace('.txt', '.jpg')
		image = Image.open(os.path.join(data_dir, stage, image_name))
		with open(os.path.join(data_dir, stage, label_file), 'r') as f:
			lines = f.readlines()
		try:
			for i, line in enumerate(lines):
				line = line.strip().split('\t')
				text_type = line[2]
				text_content = line[1]
				text_loc = [int(t) for t in line[0].split(' ')]
				xmin, xmax = min(text_loc[::2]), max(text_loc[::2])
				ymin, ymax = min(text_loc[1::2]), max(text_loc[1::2])
				cropped_image = image.crop([xmin, ymin, xmax, ymax])
				cropped_image_name = image_name + '_{}.jpg'.format(i)
				cropped_image.save(os.path.join(to_save_dir, stage, cropped_image_name))
				rec_gt_list.append([cropped_image_name, text_content, text_type])
		except:
			print('corrupted file: ', label_file)
			continue
	with open(os.path.join(to_save_dir, stage + '_gt.txt'), 'w') as f:
		for item in rec_gt_list:
			f.write('\t'.join(item) + '\n')
	print('{} {}'.format(stage, len(rec_gt_list)))
print('Finished')


