import argparse
import os
import shutil
import time
import utils
import logging
import torch
import model_loader
from tqdm import tqdm
from Evaluation_Matric import F1, AverageMeter, accuracy
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
parser.add_argument('--threshold', default=0.5, type=float,
                    help='the threshold of the prediction')



def main():
    global args
    args = parser.parse_args()
    
    best_F1 = (1e-4, 1e-4)
    
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
    logging.warning("Loading Model")
    model, version = model_loader.loadModel(args.network, params, pretrained = True)
    logging.warning("Model Loaded")
    
    model.cuda()
    # define loss function and optimizer
    loss = model_loader.UnevenWeightBCE_loss
    optimizer = torch.optim.Adam(model.parameters(), params.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=params.weight_decay, amsgrad=False)

    cudnn.benchmark = True
    
    
    #resume checkpoint
    checkpointfile = os.path.join(json_path, args.network + version + '.pth.tar')
    if args.resume:
        if os.path.isfile(checkpointfile):
            logging.info("Loading checkpoint {}".format(checkpointfile))
            checkpoint = torch.load(checkpointfile)
            args.start_epoch = checkpoint['epoch']
            best_F1 = checkpoint['best_F1']
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

    val_loader = dataloaders['val']

    logging.info(model)

    validate(val_loader, model, loss, threshold = args.threshold)

    for epoch in range(params.start_epoch, params.epochs):

        # train for one epoch
        train(train_loader, model, loss, optimizer, epoch, threshold = args.threshold)

        # evaluate on validation set
        val_result = validate(val_loader, model, loss, threshold = args.threshold)

        # remember best F1 and save checkpoint
        is_best = (2*(val_result[1][0]*val_result[2][0])/(val_result[1][0]+val_result[2][0])) > (2*(best_F1[0]*best_F1[1])/(best_F1[0]+best_F1[1]))
        best_F1 = max((val_result[1][0],val_result[2][0]), best_F1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_F1': best_F1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, path=json_path, filename=checkpointfile, version=version, network=args.network)
        if is_best:
            save_to_ini(params, args.model_dir, args.network, version, val_result, threshold = args.threshold)
    validate(val_loader, model, loss, threshold = args.threshold)
        

def train(train_loader, model, loss, optimizer, epoch, threshold = 0.5):
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
            prec, diab_pos, glau_pos = accuracy(output.data, label, threshold = threshold)
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
        
        diabF1 = F1(diab)
        glauF1 = F1(glau)
        logging.warning('Epoch: [{0}][{1}/{2}]\n'
              '    Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
              '    Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
              '    Loss {loss.val:.4f} ({loss.avg:.4f})\n'
              '    Accuracy Diabetes@ {acc[0].avg:.3f}({acc[0].avg:.4f})\n'
              '        Diabetes F1 {diabF1[0]:.4f}({diabF1[0]:.4f})\n'
              '            Diabetes recall {diabF1[1]:.4f}({diabF1[1]:.4f})\n'
              '            Diabetes precision {diabF1[2]:.4f}({diabF1[2]:.4f})\n'
              '    Accuracy Glaucoma@ {acc[1].avg:.3f}({acc[1].avg:.4f})\n'
              '        Glaucoma F1 {glauF1[0]:.4f}({glauF1[0]:.4f})\n'
              '            Glaucoma recall {glauF1[1]:.4f}({glauF1[1]:.4f})\n'
              '            Glaucoma precision {glauF1[2]:.4f}({glauF1[2]:.4f})\n'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, acc = acc, diabF1 = diabF1, glauF1 = glauF1))


def validate(val_loader, model, loss, threshold = 0.5):
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
        prec, diab_pos, glau_pos = accuracy(output.data, label, threshold = threshold)
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

    diabF1 = F1(diab)
    glauF1 = F1(glau)
    logging.warning('Test: [{0}/{1}]\n'
          '    Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
          '    Loss {loss.val:.4f} ({loss.avg:.4f})\n'
          '    Accuracy Diabetes@ {acc[0].avg:.3f}({acc[0].avg:.4f})\n'
          '        Diabetes F1 {diabF1[0]:.4f}({diabF1[0]:.4f})\n'
          '            Diabetes recall {diabF1[1]:.4f}({diabF1[1]:.4f})\n'
          '            Diabetes precision {diabF1[2]:.4f}({diabF1[2]:.4f})\n'
          '    Accuracy Glaucoma@ {acc[1].avg:.3f}({acc[1].avg:.4f})\n'
          '        Glaucoma F1 {glauF1[0]:.4f}({glauF1[0]:.4f})\n'
          '            Glaucoma recall {glauF1[1]:.4f}({glauF1[1]:.4f})\n'
          '            Glaucoma precision {glauF1[2]:.4f}({glauF1[2]:.4f})\n'.format(
           i, len(val_loader), batch_time=batch_time, loss=losses, acc = acc, diabF1 = diabF1, glauF1 = glauF1))


    return acc, diabF1, glauF1


def save_checkpoint(state, is_best, path, filename, version, network):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, network + version + '_model_best.pth.tar') )

def save_to_ini(params, path, network, version, val_result, threshold):
    config = utils.ConfigParser()
    section_name = network + str(version)
    config_name = os.path.join(path, 'BestCompare.ini')
    if os.path.isfile(config_name):
        config.read(config_name)
        if config.has_section(section_name):
            config.read(section_name)
        else:
            config.add_section(section_name)
    else:
        config.add_section(section_name)
    config.set(section_name, 'Diabetes Accuracy', str(val_result[0][0].avg))
    config.set(section_name, '    Diabetes F1', str(val_result[1][0]))
    config.set(section_name, '        Diabetes recall', str(val_result[1][1]))
    config.set(section_name, '        Diabetes precision', str(val_result[1][2]))
    
    config.set(section_name, 'Glaucoma Accuracy', str(val_result[0][1].avg))
    config.set(section_name, '    Glaucoma F1', str(val_result[2][0]))
    config.set(section_name, '        Glaucoma recall', str(val_result[2][1]))
    config.set(section_name, '        Glaucoma precision', str(val_result[2][2]))
    config.set(section_name, 'learning_rate', str(params.learning_rate))
    config.set(section_name, 'weight_decay', str(params.weight_decay))
    config.set(section_name, 'threshold', str(threshold))
    config.write(open(config_name, 'w+'))
    return 0

if __name__ == '__main__':
    main()
