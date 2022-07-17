from models.torch_models import *
import sklearn.metrics as metrics
import sys
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime
from utils import get_n_config, draw_molecule_2d
from dataset_utils import get_dataloaders
import numpy as np


def get_grad_norm(model):
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(params) == 0:
        norm = 0
    else:
        norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).cpu() for p in params]), 2.0).item()

    return norm


def set_lr(schedulers, optimizer, lr_schedule, learning_rate, max_lr, err_tr, hit_max_lr):
    if lr_schedule:
        lr = optimizer.param_groups[0]['lr']
        if lr > 1e-4:
            schedulers[0].step(np.mean(np.asarray(err_tr)))  # plateau scheduler

        if not hit_max_lr:
            if lr <= max_lr:
                schedulers[1].step()
            else:
                hit_max_lr = True
        elif hit_max_lr:
            if lr > learning_rate:
                schedulers[2].step()  # start reducing lr
    lr = optimizer.param_groups[0]['lr']
    return optimizer, lr


def computeF1(matrix, nClasses):
    truePositive = [matrix[i, i] for i in range(nClasses)]
    falsePositive = [np.sum(matrix[i, :]) - matrix[i, i] for i in range(nClasses)]
    falseNegative = [np.sum(matrix[:, i]) - matrix[i, i] for i in range(nClasses)]

    accuracy = np.sum(matrix.diagonal()) / np.sum(matrix)
    recall = np.asarray([truePositive[i] / (truePositive[i] + falsePositive[i]) for i in range(nClasses)])
    precision = np.asarray([truePositive[i] / (truePositive[i] + falseNegative[i]) for i in range(nClasses)])
    F1 = np.asarray([2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(nClasses)])

    return accuracy, np.average(np.nan_to_num(precision, nan=0)), np.average(np.nan_to_num(recall, nan=0)), np.average(np.nan_to_num(F1, nan=0))


def computeTopXAccuracy(config, probs, targets, X=5):
    # this actually computes the 'true positive rate' true positive / all positives
    correct_counter = np.zeros(config.dataDims['output classes'][0], dtype='uint64')
    incorrect_counter = np.zeros_like(correct_counter)

    for i in range(len(probs)):
        topXPredictions = np.argpartition(probs[i], -X)[-X:]  # not sorted
        if targets[i] in topXPredictions:
            correct_counter[targets[i]] += 1
        else:
            incorrect_counter[targets[i]] += 1

    overallTopXAccuracy = correct_counter.sum() / (correct_counter.sum() + incorrect_counter.sum())
    byGroupTopXAccuracy = np.zeros(len(correct_counter))
    for i in range(len(correct_counter)):
        byGroupTopXAccuracy[i] = correct_counter[i] / (correct_counter[i] + incorrect_counter[i])

    return overallTopXAccuracy, byGroupTopXAccuracy


def checkConvergence(record, history, convergence_eps):
    """
    check if we are converged
    condition: test loss has increased or levelled out over the last several epochs
    :return: convergence flag
    """

    converged = False
    if type(record) == list:
        record = np.asarray(record)

    if len(record) > (history + 2):
        if all(record[-history:] > np.amin(record)):
            converged = True
            print("Model converged, target diverging")

        criteria = np.var(record[-history:]) / np.abs(np.average(record[-history:]))
        print('Convergence criteria at {:.3f}'.format(np.log10(criteria)))
        if criteria < convergence_eps:
            converged = True
            print("Model converged, target stabilized")

    return converged


def save_model(model, optimizer):
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/model_ckpt')


def load_model(config, model, optimizer):
    '''
    Check if a checkpoint exists for this model - if so, load it
    :return:
    '''
    checkpoint = torch.load('ckpts/model_ckpt')

    if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config.device == 'cuda':
        # model = nn.DataParallel(self.model) # enables multi-GPU training
        print("Using ", torch.cuda.device_count(), " GPUs")
        model.to(torch.device("cuda:0"))
        for state in optimizer.state.values():  # move optimizer to GPU
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    return model, optimizer
