"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import pickle


class AnnotationTransform(object):
    """
    Transforms a widerface annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self):
        pass
   
    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        num = int(target[0])
        # print('num of annos: ' + str(num))
        res = []
        for i in range(num):
            if 4+i*4 >= len(target):
                continue
            xmin = int(target[1+i*4])
            ymin = int(target[2+i*4])
            xmax = int(target[3+i*4]) + xmin
            ymax = int(target[4+i*4]) + ymin
            if int(target[3+i*4]) == 0 or int(target[4+i*4]) ==0:
                continue

            elif int(target[3+i*4]) < 0:
                tmp = xmin
                xmin = xmax
                xmax = tmp
            elif int(target[4+i*4]) < 0:
                tmp = ymin
                ymin = ymax
                ymax = tmp

            res.append([xmin/float(width),ymin/float(height),xmax/float(width),ymax/float(height),0])
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class Detection(data.Dataset):
    """
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
    """
    def __init__(self, anno_file, transform=None, target_transform=None,
                 dataset_name='Yuncong'):
        self.anno_file = anno_file
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self.annotation = list()
        self.counter = 0
        # self.counterzsd = 0
        for line in open(self.anno_file,'r'):
            if line is None or len(line)<=1:
                continue
            
            line0 = line.strip().split(' ')
            if line0[0] == 'Part_B/train_data/IMG_108.jpg':
                continue
            if line0[0] == 'Part_A/train_data/IMG_126.jpg':
                continue
            if line0[0] == 'Part_A/train_data/IMG_24.jpg':
                continue
            if line0[0] == 'Part_/train_data/IMG_64.jpg':
                continue
            if line0[0] == 'Part_A/train_data/IMG_84.jpg':
                continue
            if line0[0] == 'Part_A/train_data/IMG_210.jpg':
                continue
            if line0[0] == 'Part_A/train_data/IMG_257.jpg':
                continue
            if line0[0] == 'Part_A/train_data/IMG_277.jpg':
                continue
            if line0[1] == '0':
                continue
            if line0[0].startswith('train'):
                filename = './yuncong_data/our/' + line0[0]
            else:
                filename = './yuncong_data/' + line0[0]
            # self.counterzsd += 1
            self.ids.append(filename)
            line1 = [line0[indexx] for indexx,ix in enumerate(line0) if (indexx-2)%5!=0 and indexx > 0]
            self.annotation.append(line1)
            if line0[0].startswith('Part_A'):
                self.ids.append(filename)
                line1 = [line0[indexx] for indexx, ix in enumerate(line0) if (indexx - 2) % 5 != 0 and indexx > 0]
                self.annotation.append(line1)
            if line0[0].startswith('Part_B'):
                self.ids.append(filename)
                line1 = [line0[indexx] for indexx, ix in enumerate(line0) if (indexx - 2) % 5 != 0 and indexx > 0]
                self.annotation.append(line1)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        #print(im, gt)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = self.annotation[index]
        img = cv2.imread(img_id)
        if (img == []):
            print('resize,please check line')
            print('img_id: '+ str(img_id))
            os.remove(img_id)

        else:
            height, width, channels = img.shape
            if(channels== 1):
                print('resize,please check line')
                print('img_id: ' + str(img_id))


        if self.target_transform is not None:
            target = self.target_transform(target, width, height) 
        #print('target.shape = ' + str(np.array(target).shape))

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        
    def _resize_bbox(line, time=1.2):
        assert len(line)%4 == 1, 'the line is in wrong format!'+str(len(line))     
        npline = np.array(line[1:]).reshape(-1,4).astype('float64')
        npline[:,0] = npline[:,0] - (time-1)*npline[:,2]/4
        npline[:,1] = npline[:,1] - (time-1)*npline[:,3]/4
        npline[:,2] = npline[:,2]*time
        npline[:,3] = npline[:,3]*time
        return [line[0]]+npline.reshape(1,-1).squeeze().astype('uint32').tolist()

    def _lift_bbox(line, filename, lift=0.2, half=True):
        assert len(line)%4 == 1, 'the line is in wrong format!'+str(len(line))
        im = cv2.imread(filename)
        assert im is not None, 'the image to lift box is none!'
        h = im.shape[0]
        npline = np.array(line[1:]).reshape(-1,4).astype('float64')
        linemask = npline[:,1] > h/2
        lowline = npline[linemask,:]
        upline = npline[~linemask,:]
        lowline[:,1] = lowline[:,1]-lowline[:,3]*0.3
        return [line[0]]+np.vstack((lowline,upline)).reshape(1,-1).squeeze().astype('uint32').tolist()
        
        
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
