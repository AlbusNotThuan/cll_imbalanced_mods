import os
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
import argparse
from imb_cll.dataset.dataset import prepare_dataset, CLCustomDataset
from imb_cll.utils.utils import adjust_learning_rate, AverageMeter, compute_metrics_and_record
from imb_cll.utils.metrics import accuracy

num_classes = 10
num_workers = 4
device = "cuda"

def get_resnet18():
    # resnet = torchvision.models.resnet18(weights=None)
    resnet = torchvision.models.resnet18()
    # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet

def get_modified_resnet18():
    resnet = torchvision.models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet

def validate(model, dataloader, eval_n_epoch, epoch):

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    all_preds = list()
    all_targets = list()

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    with torch.no_grad():
        for i, (inputs, labels, true_labels) in enumerate(dataloader):
            inputs, labels, true_labels = inputs.to(device), labels.to(device), true_labels.to(device)
            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, true_labels, topk=(1, 5))
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(true_labels.cpu().numpy())
            loss = criterion(outputs, true_labels).mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            if i % eval_n_epoch == 0:
                output = ('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch,
                        i,
                        len(dataloader),
                        loss=losses,
                        top1=top1,
                        top5=top5))
                print(output)

        cls_acc_string = compute_metrics_and_record(all_preds,
                                all_targets,
                                losses,
                                top1,
                                top5,
                                flag='Testing')
        
    if cls_acc_string is not None:
            return top1.avg, cls_acc_string
    else:
        return top1.avg

def get_dataset_T(dataset):
    dataset_T = np.zeros((num_classes,num_classes))
    class_count = np.zeros(num_classes)
    for i in range(len(dataset)):
        dataset_T[dataset.dataset.ord_labels[i]][dataset.dataset.targets[i]] += 1
        class_count[dataset.dataset.ord_labels[i]] += 1
    for i in range(num_classes):
        dataset_T[i] /= class_count[i]
    return dataset_T

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_cl_data(x, y, ytrue, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    for i in range(batch_size):
        while(True):
            j = random.randint(0, batch_size - 1)
            if y[i] != ytrue[j] and y[j] != ytrue[i]:
                mixed_x[i] = lam * x[i] + (1 - lam) * x[j]
                y_a[i], y_b[i] = y[i], y[j]
                break
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a = y_a.squeeze()
    y_b = y_b.squeeze()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(args):
    dataset_name = args.dataset_name
    algo = args.algo
    model = args.model
    lr = args.lr
    seed = args.seed
    data_aug = True if args.data_aug.lower()=="true" else False
    eval_n_epoch = args.evaluate_step
    batch_size = args.batch_size
    epochs = args.n_epoch
    n_weight = args.weighting
    mixup = args.mixup

    best_acc1 = 0.

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if data_aug:
        print("Use data augmentation.")

    if dataset_name == "cifar10":
        trainset, testset, input_dim, num_classes = prepare_dataset(args.dataset_name, args.max_train_samples, 
                                                args.multi_label, data_aug, args.imb_type, args.imb_factor)
    else:
        raise NotImplementedError

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if args.model == "resnet18":
        model = get_resnet18().to(device)
    elif args.model == "m-resnet18":
        model = get_modified_resnet18().to(device)
    else:
        raise NotImplementedError

    # with tqdm(range(epochs), unit="epoch") as tepoch:
        # tepoch.set_description(f"lr={lr}")
    tepoch = args.n_epoch
    for epoch in range(0, tepoch):
        training_loss = 0.0
        model.train()
        weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
        # weights = torch.tensor([0.69296145, 0.8703214, 0.97548631, 1.03682939, 1.07463482, 1.0961549, 1.10995745, 1.1176148, 1.12220554, 1.1252645])
        weights = weights ** n_weight
        weights = weights.to(device)

        # for confusion matrix
        all_preds = list()
        all_targets = list()
        cl_samples = list()

        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        if epoch > 250:
            learning_rate = adjust_learning_rate(epochs, epoch, lr)
            # learning_rate = lr
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            for inputs, labels, true_labels in trainloader:
                inputs, labels, true_labels = inputs.to(device), labels.to(device), true_labels.to(device)
            
            # for inputs, labels in trainloader:
            #     inputs, labels = inputs.to(device), labels.to(device)

                # Two kinds of output
                optimizer.zero_grad()
                outputs = model(inputs)

                if mixup:
                    # Mixup Data
                    # _input_mix, target_a, target_b, lam = mixup_data(inputs, labels)
                    _input_mix, target_a, target_b, lam = mixup_cl_data(inputs, labels, true_labels, device)
                    output_mix = model(_input_mix)
                    max_prob_mix, target_mix = torch.max(output_mix, dim=1)

                    if algo == "scl-exp":
                        # outputs = F.softmax(outputs, dim=1)
                        output_mix = F.softmax(output_mix, dim=1)
                        # labels = labels.squeeze()
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        # loss = -F.nll_loss(outputs.exp(), labels)
                        loss = lam * (-F.nll_loss(output_mix.exp(), target_a, weights)) + (1 - lam) * (-F.nll_loss(output_mix.exp(), target_b, weights))
                    
                    # elif algo == "scl-fwd":
                    #     q = torch.mm(F.softmax(outputs, dim=1), Q) + 1e-6
                    #     loss = F.nll_loss(q.log(), labels.squeeze())
                    #     loss.backward()
                    
                    elif algo == "scl-nl":
                        # p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                        p = (1 - F.softmax(output_mix, dim=1) + 1e-6).log()
                        # labels = labels.squeeze()
                        # loss = F.nll_loss(p, labels)
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        loss = lam * F.nll_loss(p, target_a, weights) + (1 - lam) * F.nll_loss(p, target_b, weights)
                        
                    else:
                        raise NotImplementedError
                else:
                    if algo == "scl-exp":
                        outputs = F.softmax(outputs, dim=1)
                        labels = labels.squeeze()
                        loss = -F.nll_loss(outputs.exp(), labels)
                    
                    # elif algo == "scl-fwd":
                    #     q = torch.mm(F.softmax(outputs, dim=1), Q) + 1e-6
                    #     loss = F.nll_loss(q.log(), labels.squeeze())
                    
                    elif algo == "scl-nl":
                        p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                        labels = labels.squeeze()
                        loss = F.nll_loss(p, labels)

                    else:
                        raise NotImplementedError

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                # Calcuate the number of sample in each class
                samples_class = labels.cpu().numpy()
                cl_samples.append(samples_class)

                # measure accuracy and record loss
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

                # Optimizer
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

                # tepoch.set_postfix(loss=loss.item())

                if i % eval_n_epoch == 0:
                    output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch,
                                i,
                                len(trainloader),
                                loss=losses,
                                top1=top1,
                                top5=top5,
                                lr=learning_rate))
                    print(output)
                    # log_training.write(output + '\n')
                    # log_training.flush()
            
            compute_metrics_and_record(all_preds,
                                    all_targets,
                                    losses,
                                    top1,
                                    top5,
                                    flag='Training')
            
            # Count the number of uncertainty samples for each class
            cl_samples = np.concatenate(cl_samples, axis=0)
            classes, class_counts = np.unique(cl_samples, return_counts=True)
            print("Total complementary labels in training dataset:", class_counts)

            # training_loss /= len(trainloader)
        
            # if (epoch+1) % eval_n_epoch == 0:
            acc1 = validate(model, testloader, eval_n_epoch, epoch)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            
            # test_acc = validate(model, testloader)
            # print("Accuracy(test)", test_acc)
            
        else:
            # learning_rate = adjust_learning_rate(epochs, epoch, lr)
            learning_rate = lr
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            for i, (inputs, labels, true_labels) in enumerate(trainloader):
                inputs, labels, true_labels = inputs.to(device), labels.to(device), true_labels.to(device)
                
                # Two kinds of output
                optimizer.zero_grad()
                outputs = model(inputs)

                if mixup:
                    # Mixup Data
                    # _input_mix, target_a, target_b, lam = mixup_data(inputs, labels)
                    _input_mix, target_a, target_b, lam = mixup_cl_data(inputs, labels, true_labels, device)
                    output_mix = model(_input_mix)
                    # import pdb
                    # pdb.set_trace()
                    max_prob_mix, target_mix = torch.max(output_mix, dim=1)

                    if algo == "scl-exp":
                        output_mix = F.softmax(output_mix, dim=1)
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        loss = lam * (-F.nll_loss(output_mix.exp(), target_a)) + (1 - lam) * (-F.nll_loss(output_mix.exp(), target_b))
                    
                    # elif algo == "scl-fwd":
                    #     q = torch.mm(F.softmax(outputs, dim=1), Q) + 1e-6
                    #     loss = F.nll_loss(q.log(), labels.squeeze())
                    #     loss.backward()
                    
                    elif algo == "scl-nl":
                        p = (1 - F.softmax(output_mix, dim=1) + 1e-6).log()
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        loss = lam * F.nll_loss(p, target_a) + (1 - lam) * F.nll_loss(p, target_b)

                    else:
                        raise NotImplementedError
                else:
                    if algo == "scl-exp":
                        outputs = F.softmax(outputs, dim=1)
                        labels = labels.squeeze()
                        loss = -F.nll_loss(outputs.exp(), labels)
                    
                    # elif algo == "scl-fwd":
                    #     q = torch.mm(F.softmax(outputs, dim=1), Q) + 1e-6
                    #     loss = F.nll_loss(q.log(), labels.squeeze())
                    
                    elif algo == "scl-nl":
                        p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                        labels = labels.squeeze()
                        loss = F.nll_loss(p, labels)

                    else:
                        raise NotImplementedError

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                # Calcuate the number of sample in each class
                samples_class = labels.cpu().numpy()
                cl_samples.append(samples_class)

                # measure accuracy and record loss
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

                # Optimizer
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

                # tepoch.set_postfix(loss=loss.item())

                if i % eval_n_epoch == 0:
                    output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch,
                                i,
                                len(trainloader),
                                loss=losses,
                                top1=top1,
                                top5=top5,
                                lr=learning_rate))
                    print(output)
                    # log_training.write(output + '\n')
                    # log_training.flush()

            compute_metrics_and_record(all_preds,
                                    all_targets,
                                    losses,
                                    top1,
                                    top5,
                                    flag='Training')
            
            # Count the number of uncertainty samples for each class
            cl_samples = np.concatenate(cl_samples, axis=0)
            classes, class_counts = np.unique(cl_samples, return_counts=True)
            print("Total complementary labels in training dataset:", class_counts)

            # training_loss /= len(trainloader)
            
            # if (epoch+1) % eval_n_epoch == 0:
            acc1 = validate(model, testloader, eval_n_epoch, epoch)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)

if __name__ == "__main__":

    dataset_list = [
        "cifar10",
        "cifar20"
    ]

    algo_list = [
        "scl-exp",
        "scl-nl",
        "scl-fwd"
    ]

    model_list = [
        "resnet18",
        "m-resnet18"
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, choices=algo_list, help='Algorithm')
    parser.add_argument('--dataset_name', type=str, choices=dataset_list, help='Dataset name', default='cifar10')
    parser.add_argument('--model', type=str, choices=model_list, help='Model name', default='resnet18')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--seed', type=int, help='Random seed', default=1126)
    parser.add_argument('--data_aug', type=str, default='false')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--evaluate_step', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--multi_label', action='store_true')
    parser.add_argument('--imb_type', type=str, default=None)
    parser.add_argument('--imb_factor', type=float, default=1.0)
    parser.add_argument('--weighting', type=int, default=1)
    parser.add_argument('--mixup', type=str, default='false')

    args = parser.parse_args()

    train(args)
