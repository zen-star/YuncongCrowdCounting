import cv2
import numpy as np
import math

image_path = './yuncong_data/'
gt_file = './yuncong_data/nice_zsd.txt'


def calc_cv2_bgr_mean_std(txtfile):
    '''
    sum1=x1  +x2  +...+xn
    sum2=x1^2+x2^2+...+xn^2
    mean=sum1/n                     -----> means=(116.625,119.167,120.135)
    var=sum2/n-mean^2
    std=sqrt(var)                   -----> std  =(65.461 , 65.567, 65.451)
    '''
    b_sum1 = 0
    g_sum1 = 0
    r_sum1 = 0
    b_sum2 = 0
    g_sum2 = 0
    r_sum2 = 0
    size_sum = 0
    with open(txtfile,'r') as fr:
        count = 0
        for line in fr.readlines():
            count += 1
            print(count)
            fields = line.strip().split(' ')
            if fields[0].startswith('train'):
                filename = image_path + 'our/' + fields[0]
            else:
                filename = image_path + fields[0]

            image = cv2.imread(filename)
            b, g, r = cv2.split(image)
            h, w, c = image.shape
            
            b_sum1 += b.sum()
            g_sum1 += g.sum()
            r_sum1 += r.sum()
            b_sum2 += (b.astype('uint16')*b).sum()
            g_sum2 += (g.astype('uint16')*g).sum()
            r_sum2 += (r.astype('uint16')*r).sum()            
            size_sum += h*w

    sum1 = [b_sum1, g_sum1, r_sum1]
    mean = [s1/size_sum for s1 in sum1]
    print('mean={}'.format(tuple(mean)))
    # mean = [117,119,120]
    sum2 = [b_sum2, g_sum2, r_sum2]
    std = [math.sqrt(s2/size_sum-m**2) for m,s2 in zip(mean,sum2)]
    print('std={}'.format(tuple(std)))


if __name__ == '__main__':
    calc_cv2_bgr_mean_std(gt_file)
            
