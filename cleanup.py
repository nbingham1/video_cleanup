#!/usr/bin/python

import imageio
import sys
import os
import math
import cv2
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

images = []
transforms = []
trajectory = [[0.0, 0.0, 0.0]]
corners = []
greys = []
tdx = 0
tdy = 0
tda = 0
scene = 0

def myMean(lst):
	l = float(len(lst))
	x = sum([p[0] for p in lst])
	y = sum([p[1] for p in lst])
	return [x/l, y/l]

def mySTD(lst):
	if len(lst) <= 1:
		return 0
	else:
		mean = myMean(lst)
		std = 0
		l = float(len(lst))
		for e in lst:
			dx = (e[0] - mean[0])
			dy = (e[1] - mean[1])
			std += dx*dx + dy*dy
		return std/l

def myVar(lst, left, right, top, bottom):
	width = 5
	height = 5

	buckets = [0 for _ in range(0, width*height)]

	for i in lst:
		x = int(width*((i[0] - left)/(right-left+0.01)))
		y = int(height*((i[1] - bottom)/(top-bottom+0.01)))
		buckets[y*height + x] += 1

	mean = float(len(lst))/float(len(buckets))

	myvar = 0
	for i in buckets:
		myvar += (i - mean)*(i - mean)
	myvar /= float(len(buckets))
	return myvar

def getVel(pos0, pos1):
	return [[x[1][0] - x[0][0], x[1][1] - x[0][1]] for x in zip([p[0] for p in pos0.tolist()], [p[0] for p in pos1.tolist()])]

def magVel(vel):
	mags = [v[0]*v[0] + v[1]*v[1] for v in vel]
	return sum(mags)/float(len(mags))

def getQual(vel, idx, sel):
	qual = 0
	idset = list(set(idx))
	frame = myMean(np.array(vel)[idx == sel].tolist())
	if len(idset) > 1:
		for k in idset:
			if k != sel:
				v = myMean(np.array(vel)[idx == k].tolist())
				qual += (v[0] - frame[0])*(v[0] - frame[0]) + (v[1] - frame[1])*(v[1] - frame[1])
		qual /= float(len(idset)-1)
	return qual


def pickBack(dev, vel, idx, sz):
	#args = np.argsort(np.array(sz)).tolist()
	#idset = list(set(idx))
	#qual = []
	#for k in idset:
	#	qual.append(getQual(vel, idx,k))
	#result = np.argmin(np.array(qual))
	#print sz[result]
	#return result
	args = np.argsort(np.array(sz)).tolist()
	results = []
	i = 0
	while i < len(args) and (i == 0 or sz[args[-(i+1)]] > 40):
		results.append(args[-(i+1)])
		i += 1
	return results[np.argmin([dev[r] for r in results])]

def mask_img(img):
	return np.array([[1 if img[y,x] > 100 else 0
		for x in range(0, img.shape[1])]
		for y in range(0, img.shape[0])], dtype=np.uint8)

def fixImg(img):
	#grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	#mask = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 30)
	#mask = cv2.dilate(mask, kernel, iterations=1)
	#mask = cv2.erode(mask, kernel, iterations=1)
	#mask = cv2.dilate(mask, kernel, iterations=2)
	#result = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
	#cv2.imshow("img", result)
	#cv2.waitKey()
	return img

def featureMask(img):
	ekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	skernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	grey2 = cv2.erode(grey, ekernel, iterations=5)
	grey2 = cv2.GaussianBlur(grey2, (11,11), 0)
	grey2 = cv2.dilate(grey2, ekernel, iterations=5)

	grey = cv2.dilate(grey, ekernel, iterations=5)
	grey = cv2.GaussianBlur(grey, (11,11), 0)
	grey = cv2.erode(grey, ekernel, iterations=5)

	grey2 = cv2.pyrDown(grey2)
	grey2 = cv2.pyrDown(grey2)
	grey = cv2.pyrDown(grey)
	grey = cv2.pyrDown(grey)

	mask = cv2.adaptiveThreshold(grey2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 8)
	mask = cv2.dilate(mask, ekernel, iterations=1)
	grey = cv2.inpaint(grey, mask, 1, cv2.INPAINT_TELEA)
	
	#grey = cv2.filter2D(grey, -1, skernel)
	#cv2.imshow("img", mask)
	#cv2.waitKey()
	cv2.imshow("img", grey)
	cv2.waitKey()
	return grey

if len(sys.argv) > 1: 
	fname = sys.argv[1]
	bname = os.path.splitext(fname)[0]

	reader = imageio.get_reader(fname)
	fps = reader.get_meta_data()['fps']

	for i, cur_im2 in enumerate(reader):
		cur_im = fixImg(cur_im2)
		cur_grey = featureMask(cur_im)

		if not greys:
			greys.append(cur_grey)
			corners.append(cv2.goodFeaturesToTrack(greys[-1], 250, 0.01, 30, blockSize=7))#, mask=mask_img(greys[-1]), gradientSize=3))

		left = 0
		right = cur_im.shape[1]
		top = cur_im.shape[0]
		bottom = 0

		tf = []
		offset = []
		quality = []
		step = 10
		for j in range(1, 21):
			pos0 = corners[-j]
			pos1, status, err = cv2.calcOpticalFlowPyrLK(greys[-j], cur_grey, pos0, None)

			tmp = np.array([pos0[k] for k, stat in enumerate(status) if stat[0] == 1])
			pos0 = tmp
			tmp = np.array([pos1[k] for k, stat in enumerate(status) if stat[0] == 1])
			pos1 = tmp

			if len(pos0) > 1:
				vel = getVel(pos0, pos1)

				#print vel

				dist = sch.distance.pdist(vel)
				hier = sch.linkage(dist, method='complete')
				#print dist.min()
				#print dist.max()
				idx = sch.fcluster(hier, dist.min() + (dist.max() - dist.min())*0.01+0.001, 'distance')
				#print dist.min() + (dist.max() - dist.min())*0.01+0.001
				#print idx
				idset = list(set(idx))

				clusters = [[p[0] for p in pos0[idx == k].tolist()] for k in idset]
				sz = [len(c) for c in clusters]
				dev = [myVar(c, left, right, top, bottom) for c in clusters]
				#print dev
				back0 = idset[pickBack(dev, vel, idx, sz)]
				#print myMean(np.array(vel)[idx == back0].tolist())
				#back1 = idset[np.argmax(dev)]
				#print [max(dev), max(sz)]
				#print zip(dev, sz)[np.argmax(sz)]
				#print zip(dev, sz)[np.argmax(dev)]
				#print()
				#fig = plt.figure(figsize=(40, 40))
				#dn = sch.dendrogram(hier)
				#plt.show()

				back_pos0 = pos0[idx == back0]
				back_pos1 = pos1[idx == back0]

				if back_pos1.shape[0] > 0:
					tmp = cv2.estimateRigidTransform(back_pos0, back_pos1, False)
					#print back_pos0
					#print tmp
					if tmp is not None:
						tf.append([tmp[0,2], tmp[1, 2], math.atan2(tmp[1,0], tmp[0,0])])
						#for k in idset:
						#	v = myMean(np.array(vel)[idx == k].tolist())
						#	velqual += v[0]*v[0] + v[1]*v[1]
						#if len(idset) > 0:
						#	velqual /= float(len(idset))

						quality.append(getQual(vel, idx, back0))
						offset.append(j-1)

				if len(quality) > 0:
					print quality[-1]
				if (j == 1 and len(quality) > 0 and quality[-1] < 2000) or j+step >= len(greys):
					break

		if len(tf) > 0:
			index = np.argmin(np.array(quality))
			tf = tf[index]
			offset = offset[index]
			print (i, quality[index])
		else:
			tf = None

		if tf is not None:	
			if offset > 0:
				del transforms[-offset:]
				del trajectory[-offset:]
				del greys[-offset:]
				del corners[-offset:]
				del images[-offset:]
				tdx = trajectory[-1][0]
				tdy = trajectory[-1][1]
				tda = trajectory[-1][2]

			tdx += tf[0]
			tdy += tf[1]
			tda += tf[2]

			transforms.append(tf)
			trajectory.append([tdx, tdy, tda])
			
			#print [tdx, tdy, tda]

			greys.append(cur_grey)
			corners.append(cv2.goodFeaturesToTrack(greys[-1], 250, 0.01, 30, blockSize=7))#, mask=mask_img(greys[-1]), gradientSize=3))
		elif len(images) > 0:
			print "Scene Break"
			writer = imageio.get_writer(bname + "_" + str(scene) + ".mp4", fps=fps)
			
			npim = np.array(images[0])
			border = math.sqrt(npim.shape[1]*npim.shape[1]/4 + npim.shape[0]*npim.shape[0]/4)

			right = max([-4*x[0] + npim.shape[1]/2.0 + border*math.cos(abs(x[2]) - math.pi/4.0) for x in trajectory])
			left = min([-4*x[0] + npim.shape[1]/2.0 - border*math.cos(abs(x[2]) - math.pi/4.0) for x in trajectory])
			top = max([4*x[1] - npim.shape[0]/2.0 + border*math.cos(abs(x[2]) - math.pi/4.0) for x in trajectory])
			bottom = min([4*x[1] - npim.shape[0]/2.0 - border*math.cos(abs(x[2]) - math.pi/4.0) for x in trajectory])

			#print [left, right, bottom, top]

			width = int((right - left + 15)/16)*16
			height = int((top - bottom + 15)/16)*16
			#print [width, npim.shape[1], height, npim.shape[0]]
			shape = tuple((width, height))

			print len(images)
			print shape

			cumulative = np.zeros(tuple((height, width, 3)), np.uint8)
			for j, im in enumerate(images):
				npim = np.array(im)
				
				offx = -4*trajectory[j][0] + left
				offy = -4*trajectory[j][1] + top
				dx = math.cos(-trajectory[j][2])
				dy = math.sin(-trajectory[j][2])

				tf = np.zeros((2, 3))
				tf[0,0] = dx
				tf[0,1] = -dy
				tf[1,0] = dy
				tf[1,1] = dx
				tf[0,2] = offx
				tf[1,2] = offy

				cv2.warpAffine(npim, tf, tuple((width, height)), dst=cumulative, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

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

