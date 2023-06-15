import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from imb_cll.utils.metrics import shot_acc

def adjust_learning_rate(epochs, epoch, learning_rate):
    """Sets the learning rate"""
    # total 200 epochs scheme
    if epochs == 200:
        epoch = epoch + 1
        if epoch <= 5:
            learning_rate = learning_rate * epoch / 5
        elif epoch >= 180:
            learning_rate = learning_rate * 0.0001
            # learning_rate = self.cfg.learning_rate * 1
        elif epoch >= 160:
            learning_rate = learning_rate * 0.01
            # learning_rate = self.cfg.learning_rate * 1
        else:
            learning_rate = learning_rate
    # total 300 epochs scheme
    elif epochs == 300:
        epoch = epoch + 1
        if epoch <= 5:
            learning_rate = learning_rate * epoch / 5
        elif epoch > 250:
            learning_rate = learning_rate * 0.01
        elif epoch > 150:
            learning_rate = learning_rate * 0.1
        else:
            learning_rate = learning_rate
    else:
        raise ValueError(
            "[Warning] Total epochs {} not supported !".format(epochs))
    return learning_rate

def _init_optimizer(self):
    if self.cfg.optimizer == 'sgd':
        print("=> Initialize optimizer {}".format(self.cfg.optimizer))
        optimizer = optim.SGD(self.model.parameters(),
                                self.cfg.learning_rate,
                                momentum=self.cfg.momentum,
                                weight_decay=self.cfg.weight_decay)
        return optimizer
    else:
        raise ValueError("[Warning] Selected Optimizer not supported !")
    

def compute_metrics_and_record(all_preds,
                                all_targets,
                                losses,
                                top1,
                                top5,
                                flag='Training'):
    """Responsible for computing metrics and prepare string for logger"""
    # if flag == 'Training':
    #     log = self.log_training
    # else:
    #     log = self.log_testing

    # if self.cfg.dataset == 'cifar100' or self.cfg.dataset == 'tiny200':
    #     all_preds = np.array(all_preds)
    #     all_targets = np.array(all_targets)
    #     many_acc, median_acc, low_acc = shot_acc(self.cfg,
    #                                                 all_preds,
    #                                                 all_targets,
    #                                                 self.train_dataset,
    #                                                 acc_per_cls=False)
    #     group_acc = np.array([many_acc, median_acc, low_acc])
    #     # Print Format
    #     group_acc_string = '%s Group Acc: %s' % (flag, (np.array2string(
    #         group_acc,
    #         separator=',',
    #         formatter={'float_kind': lambda x: "%.3f" % x})))
    #     print(group_acc_string)
    # else:
    #     group_acc = None
    #     group_acc_string = None

    group_acc = None
    group_acc_string = None

    # metrics (recall)
    cf = confusion_matrix(all_targets, all_preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    # overall epoch output
    epoch_output = (
        '{flag} Results: Prec@1 {top1.avg:.4f} Prec@5 {top5.avg:.4f} \
        Loss {loss.avg:.6f}'.format(flag=flag,
                                    top1=top1,
                                    top5=top5,
                                    loss=losses))
    # per class output
    cls_acc_string = '%s Class Recall: %s' % (flag, (np.array2string(
        cls_acc,
        separator=',',
        formatter={'float_kind': lambda x: "%.6f" % x})))
    print(epoch_output)
    print(cls_acc_string)

    # if eval with best model, just return
    # if self.cfg.best_model is not None:
    #     return cls_acc_string

    # log_and_tf(epoch_output,
    #                 cls_acc,
    #                 cls_acc_string,
    #                 losses,
    #                 top1,
    #                 top5,
    #                 log,
    #                 group_acc=group_acc,
    #                 group_acc_string=group_acc_string,
    #                 flag=flag)
    

def log_and_tf(self,
                epoch_output,
                cls_acc,
                cls_acc_string,
                losses,
                top1,
                top5,
                log,
                group_acc=None,
                group_acc_string=None,
                flag=None):
    """Responsible for recording logger and tensorboardX"""
    log.write(epoch_output + '\n')
    log.write(cls_acc_string + '\n')

    if group_acc_string is not None:
        log.write(group_acc_string + '\n')
    log.write('\n')
    log.flush()

    # TF
    if group_acc_string is not None:
        if flag == 'Training':
            self.tf_writer.add_scalars(
                'acc/train_' + 'group_acc',
                {str(i): x
                    for i, x in enumerate(group_acc)}, self.epoch)
        else:
            self.tf_writer.add_scalars(
                'acc/test_' + 'group_acc',
                {str(i): x
                    for i, x in enumerate(group_acc)}, self.epoch)

    else:
        if flag == 'Training':
            self.tf_writer.add_scalars(
                'acc/train_' + 'cls_recall',
                {str(i): x
                    for i, x in enumerate(cls_acc)}, self.epoch)
        else:
            self.tf_writer.add_scalars(
                'acc/test_' + 'cls_recall',
                {str(i): x
                    for i, x in enumerate(cls_acc)}, self.epoch)
    if flag == 'Trainig':
        self.tf_writer.add_scalar('loss/train', losses.avg, self.epoch)
        self.tf_writer.add_scalar('acc/train_top1', top1.avg, self.epoch)
        self.tf_writer.add_scalar('acc/train_top5', top5.avg, self.epoch)
        self.tf_writer.add_scalar('lr',
                                    self.optimizer.param_groups[-1]['lr'],
                                    self.epoch)
    else:
        self.tf_writer.add_scalar('loss/test_' + flag, losses.avg,
                                    self.epoch)
        self.tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg,
                                    self.epoch)
        self.tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg,
                                    self.epoch)
    
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)