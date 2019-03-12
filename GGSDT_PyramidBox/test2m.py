import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio

from PIL import Image, ImageDraw
from pyramid_vgg import build_vgg
from pyramid import build_sfd
from layers import *
import cv2
import numpy as np
import math

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

print('Loading model..')
res1_net = build_sfd('test', 640, 2)
#res2_net = build_sfd('test', 640, 2)
vgg1_net = build_vgg('test', 640, 2)
#vgg2_net = build_vgg('test', 640, 2)

net1 = res1_net
#net2 = res2_net
net3 = vgg1_net
#net4 = vgg2_net

net1.load_state_dict(torch.load('./weights/Res50_pyramid_75000.pth'))
#net2.load_state_dict(torch.load('./weights/Res50_pbrush_30000.pth'))
net3.load_state_dict(torch.load('./weights/Vgg16_pyramid_70000.pth'))
#net4.load_state_dict(torch.load('./weights/Res50_vgg2_60000.pth'))

netlist = [net1,net3]  # multi-model testing
#net1.cuda()
#net1.eval()
print('Finished loading model!')


def detect_face(image, shrink, net=net1):
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_CUBIC)


    print('shrink:{}'.format(shrink))

    width = x.shape[1]
    height = x.shape[0]
    x = x.astype(np.float32)
    x -= np.array([117, 119, 120],dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = Variable(x.cuda(), volatile=True)

    net.priorbox = PriorBoxLayer(width,height)
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    boxes=[]
    scores = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.01:
            score = detections[0,i,j,0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            boxes.append([pt[0],pt[1],pt[2],pt[3]])
            scores.append(score)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0] / shrink
    det_ymin = boxes[:,1] / shrink
    det_xmax = boxes[:,2] / shrink
    det_ymax = boxes[:,3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0.05)[0]
    det = det[keep_index, :]
    return det


def multi_scale_test(image, max_im_shrink, net):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st, net)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image,0.75,net)))
        det_s = np.row_stack((det_s,detect_face(image,0.5 ,net)))
        det_s = np.row_stack((det_s,detect_face(image,0.25,net)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2.5, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt, net)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b,detect_face(image,1.5,net)))
    if max_im_shrink > 2:
        bt *= 2
        while bt <= max_im_shrink: # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt, net)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink, net)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def multi_scale_test_pyramid(image, max_shrink, net):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25, net)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [0.5, 1.25, 1.75, 2, 2.25, 2.75, 3]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i], net)
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b



def flip_test(image, shrink, net):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink, net)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.4)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:800, :]
    return dets


def write_to_txt(f, det):
    f.write(test_img_path)
    
    f.write(str((det[:,4]>=0.1).sum())+'\n')
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        if score < 0.1:
            break
        if xmin < 0:
            xmin = 0.01
        if ymin < 0:
            ymin = 0.01
        f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.3f}\n'.format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))


if __name__ == '__main__':
    readpath = '../merged_list_test.txt'
    writepath = '../test_out/test_final_2m.txt'
    if os.path.exists(writepath):
        os.system('rm ' + writepath)
    test_counter = 0
    with open(readpath,'r') as fr:
        with open(writepath,'w') as fw:
            for line in fr.readlines():
                if test_counter < 861:
                    test_counter+=1
                    continue
                test_img_path = line
                if line.strip().endswith('/1'):
                    test_path = '../yuncong_test_set/' + 'images3/1.jpg'
                else:
                    test_path = '../yuncong_test_set/' + test_img_path.strip()
                print(test_path)
                image = cv2.imread(test_path, cv2.IMREAD_COLOR)
                if image is None:
                    image=cv2.imread(test_path.replace('jpg','JPG'),cv2.IMREAD_COLOR)
                max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5
                max_im_shrink = 5.25 if max_im_shrink > 5.25 else max_im_shrink
                shrink = max_im_shrink if max_im_shrink < 1 else 1
                det = [[0,0,1,1,0.0]]
 
                for ii,net in enumerate(netlist):
                    net.cuda()
                    net.eval()
                    det0 = detect_face(image, shrink, net)  # origin test
                    det1 = flip_test(image, shrink, net)    # flip test
                    [det2, det3] = multi_scale_test(image, max_im_shrink, net) #multi-scale test
                    det4 = multi_scale_test_pyramid(image, max_im_shrink, net)
                    det = np.row_stack((det, det0, det1, det2, det3, det4))
                    print('-'*5+' Model '+str(ii))
                    #net.cpu()

                dets = bbox_vote(det)
                
                write_to_txt(fw, dets)
                test_counter += 1
                print('One done and go on...................'+str(test_counter))
			
    print('-----------------------------> Finished <----------------------------------------')

