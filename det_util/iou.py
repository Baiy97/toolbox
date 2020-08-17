'''
iou calculation func for detection
'''

def area(box):
	w = box[2] - box[0]
	h = bpx[3] - box[1]
	return w * h


def area_of_intersection(det_box, gt_box):
	'''
	:param det_box: [x1, y1, x2, y2]  left-top point and right-bottom point
	:param gt_box: [x1, y1, x2, y2]
	:return: int
	'''
	if max(det_box[0], gt_box[0]) >= min(det_box[2], gt_box[2]):
		return 0
	if max(det_box[1], gt_box[1]) >= min(det_box[3], gt_box[3]):
		return 0
	w = min(det_box[2], gt_box[2]) - max(det_box[0], gt_box[0])
	h = min(det_box[3], gt_box[3]) - max(det_box[1], gt_box[1])
	return w * h


def area_of_union(det_box, gt_box):
	inter_area = area_of_intersection(det_box, gt_box)
	area_det = area(det_box)
	area_gt = area(gt_box)
	return area_det + area_gt - inter_area


def iou(det_box, gt_box):
	inter_area = area_of_intersection(det_box, gt_box)
	union_area = area_of_union(det_box, gt_box)
	assert inter_area <= union_area
	return inter_area / (union_area + 0.1)
