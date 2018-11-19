#!/usr/bin/python

import imageio
import sys
import os
import math
import cv2
import numpy as np

images = []
transforms = []
trajectory = [[0.0, 0.0, 0.0]]
corners = []
greys = []
tdx = 0
tdy = 0
tda = 0
scene = 0

def mask_img(img):
	return np.array([[1 if img[y,x] > 100 else 0
		for x in range(0, img.shape[1])]
		for y in range(0, img.shape[0])], dtype=np.uint8)

if len(sys.argv) > 1: 
	fname = sys.argv[1]
	bname = os.path.splitext(fname)[0]

	reader = imageio.get_reader(fname)
	fps = reader.get_meta_data()['fps']

	for i, cur_im in enumerate(reader):
		if not greys:
			greys.append(cv2.cvtColor(cur_im[0:len(cur_im)/3,:,:], cv2.COLOR_BGR2GRAY))
			corners.append(cv2.goodFeaturesToTrack(greys[-1], 200, 0.01, 30, blockSize=5, mask=mask_img(greys[-1]), gradientSize=3))

		cur_grey = cv2.cvtColor(cur_im[0:len(cur_im)/3,:,:], cv2.COLOR_BGR2GRAY)

		tf = []
		offset = []
		quality = []
		step = 10
		for i in range(1, 21):
			prev_corner = corners[-i]
			cur_corner, status, err = cv2.calcOpticalFlowPyrLK(greys[-i], cur_grey, prev_corner, None)

			prev_corner2 = np.array([prev_corner[j] for j, stat in enumerate(status) if stat[0] == 1])
			cur_corner2 = np.array([cur_corner[j] for j, stat in enumerate(status) if stat[0] == 1])

			if cur_corner2.shape[0] != 0:
				tmp = cv2.estimateRigidTransform(prev_corner2, cur_corner2, False, 10, 0.5, 10)
				if tmp is not None:
					tf.append([tmp[0,2], tmp[1, 2], math.atan2(tmp[1,0], tmp[0,0])])
					quality.append(cur_corner2.shape[0])
					offset.append(i-1)

			if (i == 1 and len(tf) > 0) or i+step >= len(greys):
				break

		if len(tf) > 0:
			index = quality.index(max(quality))
			tf = tf[index]
			offset = offset[index]
		else:
			tf = None

		if tf is not None:	
			tdx += tf[0]
			tdy += tf[1]
			tda += tf[2]

			if offset > 0:
				del transforms[-offset:]
				del trajectory[-offset:]
				del greys[-offset:]
				del corners[-offset:]
				del images[-offset:]

			transforms.append(tf)
			trajectory.append([tdx, tdy, tda])
			
			print [tdx, tdy, tda]

			greys.append(cur_grey)
			corners.append(cv2.goodFeaturesToTrack(greys[-1], 200, 0.01, 30, blockSize=5, mask=mask_img(greys[-1]), gradientSize=3))
		else:
			print "Scene Break"
			writer = imageio.get_writer(bname + "_" + str(scene) + ".mp4", fps=fps)
			
			npim = np.array(images[0])
			border = math.sqrt(npim.shape[1]*npim.shape[1]/4 + npim.shape[0]*npim.shape[0]/4)

			right = max([-x[0] + npim.shape[1]/2.0 + border*math.cos(abs(x[2]) - math.pi/4.0) for x in trajectory])
			left = min([-x[0] + npim.shape[1]/2.0 - border*math.cos(abs(x[2]) - math.pi/4.0) for x in trajectory])
			top = max([x[1] - npim.shape[0]/2.0 + border*math.cos(abs(x[2]) - math.pi/4.0) for x in trajectory])
			bottom = min([x[1] - npim.shape[0]/2.0 - border*math.cos(abs(x[2]) - math.pi/4.0) for x in trajectory])

			print [left, right, bottom, top]

			width = int((right - left + 15)/16)*16
			height = int((top - bottom + 15)/16)*16
			print [width, npim.shape[1], height, npim.shape[0]]
			shape = tuple((width, height))

			cumulative = np.zeros(tuple((height, width, 3)), np.uint8)
			for i, im in enumerate(images):
				npim = np.array(im)
				
				offx = -trajectory[i][0] + left
				offy = -trajectory[i][1] + top
				print [offx, offy]
				dx = math.cos(-trajectory[i][2])
				dy = math.sin(-trajectory[i][2])
				#for xi in range(0, npim.shape[1]):
				#	x = offx
				#	y = offy
				#	for yi in range(0, npim.shape[0]):
				#		if x >= 0 and y >= 0 and x < width and y < height:
				#			cumulative[int(y),int(x),:] = npim[yi,xi,:]
				#		else:
				#			print [x, y]
				#		x -= dy
				#		y += dx
				#	offx += dx
				#	offy += dy

				tf = np.zeros((2, 3))
				tf[0,0] = dx
				tf[0,1] = -dy
				tf[1,0] = dy
				tf[1,1] = dx
				tf[0,2] = offx
				tf[1,2] = offy

				cv2.warpAffine(npim, tf, tuple((width, height)), dst=cumulative)

				writer.append_data(cumulative)
	
			writer.close()

			scene += 1
			transforms = []
			trajectory = [[0.0, 0.0, 0.0]]
			images = []
			corners = []
			greys = []
			tdx = 0
			tdy = 0
			tda = 0

		images.append(cur_im)

		#x = float(im[0,0,0])/255.0
		#y = float(im[0,0,1])/255.0
		#z = float(im[0,0,2])/255.0

		#r = math.sqrt(x*x + y*y + z*z)
		#theta = math.acos(z/r)
		#phi = math.atan(y/x)

		#print [r, theta, phi]
		#writer.append_data(im[:, :, 1])

