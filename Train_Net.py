import argparse
import os
import shutil
import time
import utils
import logging
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)
import gc
cwd = os.getcwd()
import Net.Data_loader as Data_loader

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DATA_PATH', default='./data/ResizedData',
                    help='path to imagenet data (default: ./data/ResizedData)')
parser.add_argument('--model_dir', default='experiments', 
                    help="Directory containing params.json")
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--network', type=str,
                    help='select network to train on. (no default, must be specified)')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    
    
    
    #load json
    json_path = os.path.join(args.model_dir, args.network)
    assert os.path.exists(json_path), "Can not find Path {}".format(json_path)
    json_file = os.path.join(json_path, 'params.json')
    assert os.path.isfile(json_file), "No params.json configuration file for {} found at {}".format(args.network, json_path)
    params = utils.Params(json_file)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: 
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(json_path, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    args.distributed = args.world_size > 1

    # create model
    model, version = loadModel(args.network, params, pretrained = True)
    
    model.cuda()
    # define loss function and optimizer
    loss = nn.MultiLabelSoftMarginLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), params.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=params.weight_decay, amsgrad=False)

    cudnn.benchmark = True
    
    
    #resume checkpoint
    checkpointfile = os.path.join(json_path, args.network + version + '.pth.tar')
    if args.resume:
        if os.path.isfile(checkpointfile):
            print("=> loading checkpoint '{}'".format(checkpointfile))
            checkpoint = torch.load(checkpointfile)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpointfile, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpointfile))



    # Data loading code
    if not os.path.exists(args.data_dir+'/train_data'):
        print("==> Data directory"+args.data_dir+"does not exits")
        print("==> Please specify the correct data path by")
        print("==>     --data <DATA_PATH>")
        return
    
    dataloaders = Data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)

    train_loader = dataloaders['train']

    val_loader = dataloaders['train']

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
        }, is_best, json_path, version, args.network)
    validate(val_loader, model, loss)
    
def loadModel(netname, params, pretrained = True):
    Netpath = 'Net'
    Netfile = os.path.join(Netpath, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    return {
        'alexnet': loadAlexnet(pretrained),
        'densenet': loadDensenet(pretrained, params),
    }[netname]
    
def loadAlexnet(pretrained):
    import Net.alexnet
    return Net.alexnet.alexnet(pretrained = pretrained, num_classes = 2), ''
    
def loadDensenet(pretrained, params):
    import Net.densenet
    return Net.densenet.net(params.version, pretrained), params.version

def train(train_loader, model, loss, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()
    diab = (AverageMeter(), AverageMeter())
    glau = (AverageMeter(), AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()
    with tqdm(total=len(train_loader)) as t:
        for i, (datas, label, _) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
    
            input_var = torch.autograd.Variable(datas.cuda())
            label_var = torch.autograd.Variable(label.cuda()).double()
    
            # compute output
            output = model(input_var).double()
            cost = loss(output, label_var)
    
            # measure accuracy and record cost
            prec, diab, glau = accuracy(output.data, label)
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
        
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'
              'Prec@ {top.avg:.3f}({top.avg:.4f})'
          'Diabetes F1 {diabF1:.4f}({diabF1:.4f})'
          'Glaucoma F1 {glauF1:.4f}({glauF1:.4f})'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, top = top, diabF1 = F1(diab), glauF1 = F1(glau)))


def validate(val_loader, model, loss):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()
    diab = (AverageMeter(), AverageMeter())
    glau = (AverageMeter(), AverageMeter())

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
        prec, diab, glau = accuracy(output.data, label)
        losses.update(cost.data, len(datas))
        top.update(prec, datas.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    
    print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})'
          'Prec@ {top.avg:.4f}({top.avg:.4f})'
          'Diabetes F1 {diabF1:.4f}({diabF1:.4f})'
          'Glaucoma F1 {glauF1:.4f}({glauF1:.4f})'.format(
           i, len(val_loader), batch_time=batch_time, loss=losses, top = top, diabF1 = F1(diab), glauF1 = F1(glau)))


    return top.avg


def save_checkpoint(state, is_best, path, filename, version, network):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, network + version + '_model_best.pth.tar') )

def F1(T):
    return 2*(T[0].avg * T[1].avg)/(T[0].avg + T[1].avg)

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

    outputs = outputs.cpu()> 0.6
    batchsize = len(labels)
    
    acc = 0
    acc = (outputs.int()==labels.int()).cpu().sum()
    acc = acc/float(batchsize)
    True_pos_diab = ((outputs.int()[0] == 1) & (labels.int()[0] == 1)).cpu().sum()
    recall_diab = True_pos_diab/(1==labels.int()[0]).cpu().sum()
    precision_diab = True_pos_diab/(1==outputs.int()[0]).cpu().sum()
    True_pos_glau = ((outputs.int()[1] == 1) & (labels.int()[1] == 1)).cpu().sum()
    recall_glau = True_pos_glau/(1==labels.int()[1]).cpu().sum()
    precision_glau = True_pos_glau/(1==outputs.int()[1]).cpu().sum()
    
    return acc, (recall_diab, precision_diab), (recall_glau, precision_glau)


if __name__ == '__main__':
    main()
