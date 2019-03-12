import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import face, AnnotationTransform, Detection, detection_collate
from utils.augmentations import PyramidAugmentation
from layers.modules import MultiBoxLoss

from pyramid import build_sfd,SFD,SSHContext,ContextTexture  # when the backbone is resnet50

import numpy as np
import time
from layers import *


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"  # set cuda GPU index. For this code, we use two Titan XP 12G. GPU decides the batch_size and num_workers.

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='PyramidBox-res50 YUNCONG Head Counting Training')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--resume', default='local', type=str, help='local for loading ImageNet pretrained models, $RESUME_PATH for resumiing from checkpoints')  
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=2e-6, type=float, help='initial learning rate')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='./GGSDT_PyramidBox/weights/', help='Location to save checkpoint models')
parser.add_argument('--annoPath', default="./yuncong_data/nice_zsd.txt", help='Location of preprocessed data file') 
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = face

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

ssd_dim = 640  
means = (117, 119, 120)  # the means of the YUNCONG training data
num_classes = 1 + 1
batch_size = args.batch_size

max_iter = 80000
weight_decay = 0.0001
stepvalues = tuple(range(2000,80000,2000))
gamma = 0.99312
warmup = 3e-6
momentum = 0.9

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_sfd('train', 640, num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True


if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param) 


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if 'bias' in m.state_dict().keys():
            m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        if 'bias' in m.state_dict().keys():
            m.bias.data.zero_()

    if isinstance(m, nn.BatchNorm2d):
        m.weight.data[...] = 1
        m.bias.data.zero_()


for layer in net.modules():
    layer.apply(weights_init)

if not args.resume:
    print('Initializing weights...')
elif args.resume == 'local':
    print('Loading ImageNet pretrained model from local path.')
    ssd_net.load_weights('./GGSDT_PyramidBox/weights/resnet50.pth')
elif 'weights' in args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume) 
else:
    pass
    
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 3, 0.35, False, False, args.cuda)
criterion1 = MultiBoxLoss(num_classes, 0.35, True, 0, True, 3, 0.35, False, True, args.cuda)

def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset = Detection(args.annoPath, PyramidAugmentation(ssd_dim, means), AnnotationTransform())
    print('len(dataset) = ' + str(len(dataset)))
    print(dataset.__getitem__(0))
    epoch_size = len(dataset) // args.batch_size
    print('Training PyramidBox on', dataset.name)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=np.array(torch.zeros((1,)).cpu()),
            Y=np.array(torch.zeros((1, 3)).cpu()),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current PyramidBox Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=np.array(torch.zeros((1,)).cpu()),
            Y=np.array(torch.zeros((1, 3)).cpu()),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch PyramidBox Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        lr_lot = viz.line(
            X=np.array(torch.zeros((1,)).cpu()),
            Y=np.array(torch.zeros((1,1)).cpu()),
            opts=dict(
                xlabel='iteration',
                ylabel='learning-rate',
                title='Warm-up',
                legend=['lr']
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    print('data loading finished...')
    for iteration in range(args.start_iter, max_iter):
        t0 = time.time()
        try:
            if (not batch_iterator) or (iteration % epoch_size == 0):
                batch_iterator = iter(data_loader)
            adjust_learning_rate(optimizer, gamma, iteration)
            if iteration in stepvalues:
                step_index += 1
                if args.visdom:
                    viz.line(
                        X=np.array(torch.ones((1, 3)).cpu()) * epoch,
                        Y=np.array(torch.Tensor([loc_loss, conf_loss,
                            loc_loss + conf_loss]).unsqueeze(0).cpu() )/ epoch_size,
                        win=epoch_lot,
                        update='append'
                    )
                # reset epoch loss counters
                loc_loss = 0
                conf_loss = 0
                epoch += 1
            # load train data
            images, targets = next(batch_iterator)

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            # forward
            t1 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(tuple(out[0:3]), targets)
            loss_l_head, loss_c_head = criterion(tuple(out[3:6]), targets)
        
            loss = loss_l + loss_c + 0.5 * loss_l_head + 0.5 * loss_c_head
            loss.backward()
            optimizer.step()
            t2 = time.time()
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]
            if iteration % 10 == 0:
                print('front and back Timer: {} sec.' .format((t2 - t1)))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]))
                print('Loss conf: {} Loss loc: {}'.format(loss_c.data[0],loss_l.data[0]))
                print('Loss head conf: {} Loss head loc: {}'.format(loss_c_head.data[0],loss_l_head.data[0]))
                print('lr: {}'.format(optimizer.param_groups[0]['lr']))
                if args.visdom and args.send_images_to_visdom:
                    random_batch_index = np.random.randint(images.size(0))
                    viz.image(images.data[random_batch_index].cpu().numpy())
            if args.visdom:
                viz.line(
                    X=np.array(torch.ones((1, 3)).cpu()) * iteration,
                    Y=np.array(torch.Tensor([loss_l.data[0], loss_c.data[0],
                        loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu()),
                    win=lot,
                    update='append'
                )
                viz.line(
                    X=np.array(torch.ones((1,1)).cpu()) * iteration,
                    Y=np.array(torch.Tensor([optimizer.param_groups[0]['lr']]).unsqueeze(0).cpu()),
                    win=lr_lot,
                    update='append'
                )
                # hacky fencepost solution for 0th epoch plot
                if iteration == 0:
                    viz.line(
                        X=np.array(torch.zeros((1, 3)).cpu()),
                        Y=np.array(torch.Tensor([loc_loss, conf_loss,
                            loc_loss + conf_loss]).unsqueeze(0).cpu()),
                        win=epoch_lot,
                        update=True
                    )
                    viz.line(
                        X=np.array(torch.zeros((1,1)).cpu()),
                        Y=np.array(torch.Tensor([optimizer.param_groups[0]['lr']]).unsqueeze(0).cpu()),
                        win=lr_lot,
                        update=True
                    )

        except TypeError as e:
            print(e)
            print('-'*20,'jump to next iter and log.')
            continue
        except ValueError as e2:
            print(e2)
            print('='*20,'jump to next iter and log.')
            continue
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), args.save_folder + 'Res50_pyramid_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + 'Res50_pyramid_' + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    #if args.start_iter != 0:
    #    return
    if step <= 500:  # warm-up
        for param_group in optimizer.param_groups:
            param_group['lr'] = 3.33 * warmup * step
    elif step > 4000:  # exponential decay learning rate
        if step % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * gamma


if __name__ == '__main__':
    train()
