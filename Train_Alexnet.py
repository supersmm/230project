import argparse
import os
import shutil
import time
import utils
import logging
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import Alex.alexnet as alexnet

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)
import gc
cwd = os.getcwd()
import Alex.Data_loader as Data_loader

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DATA_PATH', default='./data/ResizedData',
                    help='path to imagenet data (default: ./data/ResizedData)')
parser.add_argument('--model_dir', default='experiments/greyscale', 
                    help="Directory containing params.json")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-backend', default='NCCL', type=str,
                    help='distributed backend (gloo for cpu training, NCCL for GPU training')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    args.distributed = args.world_size > 1

    # create model
    model = alexnet.alexnet(pretrained=True)
    input_size = (178,128)
    
    model.cuda()
    # define loss function and optimizer
    loss = nn.MultiLabelSoftMarginLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    cudnn.benchmark = True
    
    
    #resume checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    # Data loading code
    if not os.path.exists(args.data_dir+'/train_data'):
        print("==> Data directory"+args.data_dir+"does not exits")
        print("==> Please specify the correct data path by")
        print("==>     --data <DATA_PATH>")
        return
    
    dataloaders = Data_loader.fetch_dataloader(['train', 'val', 'test'], args.data_dir, params)

    train_loader = dataloaders['train']

    val_loader = dataloaders['train']
    test_loader = dataloaders['test']

    print(model)

    validate(val_loader, model, loss)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, loss, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, loss)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    validate(test_loader, model, loss)


def train(train_loader, model, loss, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    with tqdm(total=len(train_loader)) as t:
        for i, (datas, label) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
    
            input_var = torch.autograd.Variable(datas.cuda())
            label_var = torch.autograd.Variable(label.cuda()).double()
    
            # compute output
            output = model(input_var).double()
            cost = loss(output, label_var)
    
            # measure accuracy and record cost
            prec = accuracy(output.data, label)
            losses.update(cost.data, len(datas))
            top.update(prec, datas.size(0))
    
            # compute gradient and do SGD step
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    

            gc.collect()
            t.set_postfix(loss='{:05.3f}'.format(losses()))
            t.update()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def validate(val_loader, model, loss):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (datas, label) in enumerate(val_loader):
        input_var = torch.autograd.Variable(datas.cuda())
        label_var = torch.autograd.Variable(label.cuda()).double()

        # compute output
        output = model(input_var).double()
        cost = loss(output, label_var)

        # measure accuracy and record cost
        prec = accuracy(output.data, label)
        losses.update(cost.data, len(datas))
        top.update(prec, datas.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
               i, len(val_loader), batch_time=batch_time, loss=losses))

    print(' * Prec@ {top.avg:.3f}'
          .format(top=top))

    return top.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        
    def __call__(self):
        return self.avg


def accuracy(outputs, labels):

    outputs = np.argmax(outputs.cpu(), axis=0)
    batchsize = len(labels)
    
    acc = 0
    acc = (outputs.int()==labels.int()).cpu().sum()
    acc = acc/float(batchsize)
    return acc


if __name__ == '__main__':
    main()
