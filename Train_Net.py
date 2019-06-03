import argparse
import os
import shutil
import time
import utils
import logging
import torch
import sys
from tqdm import tqdm
import Net.loss as Customloss
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
parser.add_argument('--log', default='warning', type=str,
                    help='set logging level')

best_prec1 = (0,0)


def main():
    global args, best_prec1
    args = parser.parse_args()
    
    
    
    #load json
    json_path = os.path.join(args.model_dir, args.network)
    assert os.path.exists(json_path), "Can not find Path {}".format(json_path)
    json_file = os.path.join(json_path, 'params.json')
    assert os.path.isfile(json_file), "No params.json configuration file for {} found at {}".format(args.network, json_path)
    print(json_file)
    params = utils.Params(json_file)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: 
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(json_path, 'train.log'), args.log)



    # create model
    print("Loading Model")
    model, version = loadModel(args.network, params, pretrained = True)
    print("Model Loaded")
    
    model.cuda()
    # define loss function and optimizer
    loss = Customloss.UnvenWeightCrossEntropyLoss(weights=[0.9, 0.1]).cuda()

    optimizer = torch.optim.Adam(model.parameters(), params.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=params.weight_decay, amsgrad=False)

    cudnn.benchmark = True
    
    
    #resume checkpoint
    checkpointfile = os.path.join(json_path, args.network + version + '.pth.tar')
    if args.resume:
        if os.path.isfile(checkpointfile):
            logging.info("Loading checkpoint {}".format(checkpointfile))
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

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    dataloaders = Data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)

    train_loader = dataloaders['train']

    val_loader = dataloaders['train']

    logging.info(model)

    validate(val_loader, model, loss)

    for epoch in range(params.start_epoch, params.epochs):

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
        }, is_best, path=json_path, filename=checkpointfile, version=version, network=args.network)
    validate(val_loader, model, loss)
    
def loadModel(netname, params, pretrained = True):
    Netpath = 'Net'
    Netfile = os.path.join(Netpath, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'alexnet': 
        model, version = loadAlexnet(pretrained)
    elif 'densenet': 
        model, version = loadDensenet(pretrained, params)
    else:
        print("No model with the name {} found, please check your spelling.".format(netname))
        print("Net List:")
        print("    AlexNet")
        print("    DenseNet")
        sys.exit()
    return model, version
    
def loadAlexnet(pretrained):
    import Net.alexnet
    print("Loading AlexNet")
    return Net.alexnet.alexnet(pretrained = pretrained, num_classes = 2), ''
    
def loadDensenet(pretrained, params):
    import Net.densenet
    print("Loading DenseNet")
    return Net.densenet.net(params.version, pretrained), params.version

def train(train_loader, model, loss, optimizer, epoch):
    logging.info("Epoch {}:".format(epoch))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = (AverageMeter(), AverageMeter())
    diab = (AverageMeter(), AverageMeter(), AverageMeter())
    glau = (AverageMeter(), AverageMeter(), AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()
    with tqdm(total=len(train_loader)) as t:
        for i, (datas, label, _) in enumerate(train_loader):
            # measure data loading time
            logging.info("    Sample {}:".format(i))
            data_time.update(time.time() - end)
            logging.info("        Loading Varable")
            input_var = torch.autograd.Variable(datas.cuda())
            label_var = torch.autograd.Variable(label.cuda()).double()
    
            # compute output
            logging.info("        Compute output")
            output = model(input_var).double()
            cost = loss(output, label_var)
    
            # measure accuracy and record cost
            logging.info("        Measure accuracy")
            prec, diab_pos, glau_pos = accuracy(output.data, label)
            losses.update(cost.data, len(datas))
            for j in range(2):
                acc[j].update(prec[j], datas.size(0))
            for j in range(3):
                diab[j].update(diab_pos[j], datas.size(0))
                glau[j].update(glau_pos[j], datas.size(0))
    
            # compute gradient and do SGD step
            logging.info("        Compute gradient and do SGD step")
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
    
            # measure elapsed time
            logging.info("        Measure elapsed time")
            batch_time.update(time.time() - end)
            end = time.time()
    

            gc.collect()
            t.set_postfix(loss='{:05.3f}'.format(losses()))
            t.update()
        
        print('Epoch: [{0}][{1}/{2}]\n'
              '    Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
              '    Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
              '    Loss {loss.val:.4f} ({loss.avg:.4f})\n'
              '    Prec Diabetes@ {acc[0].avg:.3f}({acc[0].avg:.4f})\n'
              '    Prec Glaucoma@ {acc[1].avg:.3f}({acc[1].avg:.4f})\n'
              '    Diabetes F1 {diabF1:.4f}({diabF1:.4f})\n'
              '    Glaucoma F1 {glauF1:.4f}({glauF1:.4f})\n'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, acc = acc, diabF1 = F1(diab), glauF1 = F1(glau)))


def validate(val_loader, model, loss):
    logging.info("Validating")
    logging.info("Initializing measurement")
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = (AverageMeter(), AverageMeter())
    diab = (AverageMeter(), AverageMeter(), AverageMeter())
    glau = (AverageMeter(), AverageMeter(), AverageMeter())

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (datas, label, _) in enumerate(val_loader):
        logging.info("    Sample {}:".format(i))
        logging.info("        Loading Varable")
        input_var = torch.autograd.Variable(datas.cuda())
        label_var = torch.autograd.Variable(label.cuda()).double()

        # compute output
        logging.info("        Compute output")
        output = model(input_var).double()
        cost = loss(output, label_var)

        # measure accuracy and record cost
        logging.info("        Measure accuracy and record cost")
        prec, diab_pos, glau_pos = accuracy(output.data, label)
        losses.update(cost.data, len(datas))
        for j in range(2):
            acc[j].update(prec[j], datas.size(0))
        for j in range(3):
            diab[j].update(diab_pos[j], datas.size(0))
            glau[j].update(glau_pos[j], datas.size(0))

        # measure elapsed time
        logging.info("        Measure elapsed time")
        batch_time.update(time.time() - end)
        end = time.time()

    
    print('Test: [{0}/{1}]\n'
          '    Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
          '    Loss {loss.val:.4f} ({loss.avg:.4f})\n'
          '    Prec Diabetes@ {acc[0].avg:.3f}({acc[0].avg:.4f})\n'
          '    Prec Glaucoma@ {acc[1].avg:.3f}({acc[1].avg:.4f})\n'
          '    Diabetes F1 {diabF1:.4f}({diabF1:.4f})\n'
          '    Glaucoma F1 {glauF1:.4f}({glauF1:.4f})\n'.format(
           i, len(val_loader), batch_time=batch_time, loss=losses, acc = acc, diabF1 = F1(diab), glauF1 = F1(glau)))


    return acc[0].avg, acc[1].avg


def save_checkpoint(state, is_best, path, filename, version, network):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, network + version + '_model_best.pth.tar') )

def F1(T):
    #create elson to prevent divide by zero
    epslon = 1e-6
    recall = T[0].sum/(T[1].sum + epslon)
    precision = T[0].sum/(T[2].sum + epslon)
    return 2*(recall*precision)/(recall+precision)

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

    output = outputs.cpu()> 0.6
    batchsize = len(labels)

    acc_diab = 0
    acc_diab = (output.int()[:,0]==labels.int()[:,0]).cpu().sum().float()
    acc_diab = acc_diab/float(batchsize)

    acc_glau = 0
    acc_glau = (output.int()[:,1]==labels.int()[:,1]).cpu().sum().float()
    acc_glau = acc_glau/float(batchsize)

    True_pos_diab = ((output.int()[:,0] == 1) & (labels.int()[:,0] == 1)).cpu().sum()
    pos_diab = (1==labels.int()[:,0]).cpu().sum().float()
    pos_redict_diab = (1==output.int()[:,0]).cpu().sum().float()


    True_pos_glau = ((output.int()[:,1] == 1) & (labels.int()[:,1] == 1)).cpu().sum()
    pos_glau = (labels.int()[:,1] == 1).cpu().sum().float()
    pos_redict_glau = (output.int()[:,1] == 1).cpu().sum().float()

    
    return (acc_diab, acc_glau), (True_pos_diab, pos_diab, pos_redict_diab), (True_pos_glau, pos_glau, pos_redict_glau)


if __name__ == '__main__':
    main()
