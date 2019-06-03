def F1(T):
    #create elson to prevent divide by zero
    epslon = 1e-6
    recall = T[0].sum/(T[1].sum + epslon)
    precision = T[0].sum/(T[2].sum + epslon)
    return 2*(recall*precision)/(recall+precision), recall, precision

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

def accuracy(outputs, labels, threshold = 0.5):

    output = outputs.cpu()> threshold
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
