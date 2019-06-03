import os
import sys
import torch
import logging

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
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))
        logging.warning("Net List:")
        logging.warning("    AlexNet")
        logging.warning("    DenseNet")
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

def my_loss(outputs, labels):
    loss_diab = torch.sum(torch.add(2*torch.mul(labels[:, 0],torch.log(outputs[:, 0])), 0.1*torch.mul(1 - labels[:, 0],torch.log(1 - outputs[:, 0]))))
    loss_glau = torch.sum(torch.add(2*torch.mul(labels[:, 1],torch.log(outputs[:, 1])), 0.1*torch.mul(1 - labels[:, 1],torch.log(1 - outputs[:, 1]))))
    return -torch.add(loss_diab, loss_glau)
