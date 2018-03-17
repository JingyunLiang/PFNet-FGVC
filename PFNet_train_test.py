# Copyright (C) 2018 Jingyun Liang et al.
# All rights reserved.

import argparse
import os
import shutil
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import pprint
import numbers

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.nn.functional as F


# dataset preparation, self-defined transforms with rois and spp layer
from lib.car_multilable_rois import ImageFolder as car_multi
import lib.transforms_with_rois as transforms
from lib.layer_utils.roi_pooling.roi_pool import RoIPoolFunction


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='none',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10, no internel ouput: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--input-crop', default=224, type=int,
                    help='input image crop size (default: 224)')
parser.add_argument('--input-scale', default=256, type=int,
                    help='input image scale size (default: 256)')
parser.add_argument('--lr-stepsize', '--learning-rate-stepsize', default=30, type=int,
                    metavar='LR', help='learning rate stepsize')
parser.add_argument('--num-Classes', type=int,
                    help='number of dataset classes')
parser.add_argument('--maximum-Rois', dest='maximumRois', default=100, type=int,
                    help='maximum number of rois')


best_prec1 = 0
plot_statistic = {"train_loss":[],"test_loss":[],"train_acc1":[],"test_acc1":[]}


def main():
    global args, best_prec1, modelDir, log_file, plot_statistic

    args = parser.parse_args()
    args.data = 'car' # dataset name: cub car aircraft
    args.numClasses = 196 # cub 200 car 196 aircraft 100
    args.arch = 'vgg19' # backbone CNN
    args.maximumRois = 500 # number of rois
    modelDir = args.data +'_'+ args.arch +'_test' # checkpoint dir
    args.resume = os.path.join(modelDir, 'epoch-' + '15' + '-checkpoint.pth.tar') # 1,2,3, 0 for no resume checkpoint
    args.evaluate = True

    args.epochs = 20
    args.batch_size = 1
    args.lr = 1e-4
    args.lr_stepsize = 10
    args.weight_decay = 5e-4

    args.workers = 2
    args.print_freq = 10
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.distributed = args.world_size > 1

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = modelDir + "_{}.log".format(timestamp)
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)
    shutil.copy(os.path.abspath(__file__),modelDir)
    os.rename(os.path.join(modelDir,os.path.basename(__file__)),\
              os.path.join(modelDir,os.path.basename(__file__))[:-3]+"_{}.py".format(timestamp))

    printlog(args)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)


    # create model
    printlog("=> using imagenet pre-trained model '{}'".format(args.arch))
    if 'vgg' in args.arch :
        model = models.__dict__[args.arch](pretrained=True)
        model.classifier._modules['6'] = nn.Linear(model.classifier[6].in_features, args.numClasses)
        model = VggBasedNet_PFNet(originalModel = model)
    else:
        raise ValueError

    printlog(model)

    if not args.distributed:
        model = torch.nn.DataParallel(model.cuda())
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define optimizer
    params = []
    if 'vgg' in args.arch :
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'features' in key or 'conv5' in key:
                    smaller_lr = 0.1
                else:
                    smaller_lr = 1

                if 'bias' in key:
                    params += [{'params':[value],'lr':args.lr*smaller_lr, 'weight_decay': False and args.weight_decay or 0}]
                else:
                    params += [{'params':[value],'lr':args.lr*smaller_lr, 'weight_decay': args.weight_decay}]
            else:
                printlog('layer --{0}-- is fixed.'.format(key))

        optimizer = torch.optim.SGD(params, momentum=args.momentum)
    else:
        raise ValueError

    # optionally resume from a checkpoint
    if args.resume :
        printlog("=> loading specified checkpoint '{}'".format(args.resume))

        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            plot_statistic = checkpoint['loss_acc1']
            printlog("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            printlog("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # prepare data
    train_loader,val_loader,train_sampler = get_data_loader()

    # define loss
    criterion = [BinaryLogLoss().cuda(), PartAttentionLoss().cuda()]

    # model testing
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # model training
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train
        train(train_loader, model, criterion, optimizer, epoch)

        # test
        prec1 = validate(val_loader, model, criterion)

        # save checkpoint
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'loss_acc1':plot_statistic,
        })

    showPlot(plot_statistic)
    printlog('Training done, the best test_acc1 is {0} in Epoch {1}'.format(best_prec1,plot_statistic["test_acc1"].index(best_prec1)))


def train(train_loader, model, criterion, optimizer, epoch):
    """model training"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = inputs[0].cuda() # image tensor
        rois = inputs[1][0,:,:].cuda()  # rois matrix
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        rois_var = torch.autograd.Variable(rois)
        target_var = torch.autograd.Variable(target)

        # forward
        output,softMatrix = model(input_var, rois_var)
        loss = criterion[0](output, target_var)+criterion[1](softMatrix, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.print_freq:
            if i % args.print_freq == 0:
                printlog('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec_1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

    printlog('Epoch {0} \t\t\t Model {1} \t Time {2}'.format(epoch, modelDir,time.strftime("%H-%M-%S")))
    printlog('Train Loss {loss.avg:.4f}   top1 {top1.avg:.3f}  BatchTime{batch_time.avg:.3f}'
          .format(loss = losses, top1=top1, batch_time=batch_time))

    plot_statistic["train_loss"].append(losses.avg)
    plot_statistic["train_acc1"].append(top1.avg)


def validate(val_loader, model, criterion):
    """model testing"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
        input = inputs[0].cuda()
        rois = inputs[1][0,:,:].cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        rois_var = torch.autograd.Variable(rois, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output,softMatrix = model(input_var, rois_var)
        loss = criterion[0](output, target_var)+criterion[1](softMatrix, target_var)#+criterion[2](sparseSoftMatrix)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq:
            if i % args.print_freq == 0:
                printlog('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec_1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    printlog('Test  Loss {loss.avg:.4f}   top1 {top1.avg:.3f}  BatchTime{batch_time.avg:.3f}'
          .format(loss = losses, top1=top1, batch_time=batch_time))

    plot_statistic["test_loss"].append(losses.avg)
    plot_statistic["test_acc1"].append(top1.avg)
    showPlot(plot_statistic)

    return top1.avg


def save_checkpoint(state):
    """save checkpoint"""
    filename = os.path.join(modelDir, 'epoch-' + str(state['epoch']) + '-checkpoint.pth.tar')
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_stepsize))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    _, index = torch.max(target,dim=1)
    target = index

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def printlog(output):
    """print log on screen and save to .log file"""
    print(output)

    stdout_backup = sys.stdout
    logfile = open(os.path.join(modelDir,log_file),'a')
    sys.stdout = logfile
    pprint.pprint(output)
    logfile.close()
    sys.stdout = stdout_backup


def showPlot(plot_statistic):
    """plot loss and accuracy"""
    plt.clf()
    plt1 = plt.subplot(121)
    plt2 = plt.subplot(122)
    loc = ticker.MultipleLocator(base=10)
    plt1.xaxis.set_major_locator(loc)
    plt2.xaxis.set_major_locator(loc)
    plt1.plot(plot_statistic["train_loss"],label="train_loss")
    plt2.plot(plot_statistic["train_acc1"],label="train_acc1")
    plt1.plot(plot_statistic["test_loss"],label="test_loss")
    plt2.plot(plot_statistic["test_acc1"],label="test_acc1")
    plt1.legend()
    plt2.legend()
    plt.savefig(os.path.join(modelDir,'loss_acc1.png'))

def get_data_loader():
    """Data loading code"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
            transforms.Scale(args.input_crop,scaleheight=[250,350,450,550,650]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    val_transforms = transforms.Compose([
            transforms.Scale(args.input_crop,scaleheight=[250,350,450,550,650]),#79.75% for above test
            transforms.ToTensor(),
            normalize,
        ])

    if args.data == 'car':
        train_dataset = car_multi(args.data, 'trainval',transform=train_transforms)
        val_dataset = car_multi(args.data, 'test',transform=val_transforms)
    else:
        raise ValueError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_transforms)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader,val_loader,train_sampler


class VggBasedNet_PFNet(nn.Module):
    """model structure of PFNet"""
    def __init__(self, originalModel):
        super(VggBasedNet_PFNet, self).__init__()
        self.features = nn.Sequential(*list(originalModel.features)[:-1])
        self.roipooling = RoIPoolFunction(7, 7, 1. / 16.)
        self.classifier = originalModel.classifier

    def forward(self, x, rois):
        # part feature extractor
        x = self.features(x)
        x = self.roipooling(x, rois)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # two-level loss
        softMatrix = F.softmax(x, dim=1)
        x = softMatrix.sum(dim=0,keepdim=True)/args.maximumRois

        return x, softMatrix


class VggFtNet(nn.Module):
    """VGG fine-tuning"""
    def __init__(self, originalModel):
        super(VggFtNet, self).__init__()
        self.features = nn.Sequential(*list(originalModel.features))
        self.roipooling = RoIPoolFunction(7, 7, 1. / 16.)
        self.classifier = originalModel.classifier


    def forward(self, x, rois):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class BinaryLogLoss(nn.Module):
    """image loss"""
    def __init__(self):
        super(BinaryLogLoss, self).__init__()
        return

    def forward(self, input, target):
        #  t = -log(c.*(X-0.5) + 0.5) ;. x is assumed to be the
        #  probability that the attribute is active (c=+1). Hence x must be
        #  a number in the range [0,1]. This is the binary version of the`log` loss.
        return -(target.mul(input*0.9999+1e-5 -0.5)+0.5).log().sum()


class PartAttentionLoss(nn.Module):
    """part attention loss"""
    def __init__(self):
        super(PartAttentionLoss, self).__init__()
        self.lamda = 1
        return

    def forward(self, softMatrix, target):
        p_t = (target.mul(softMatrix-0.5)+0.5)*0.9999+1e-5
        return -(p_t).log().mul(\
            torch.pow(1-p_t,self.lamda)).sum()/softMatrix.size(0)*5



if __name__ == '__main__':
    main()
