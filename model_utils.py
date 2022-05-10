from torch_models import *
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import sklearn.metrics as metrics
import sys
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime
from utils import stack_dataset, unstack_dataset, get_n_config, draw_molecule_2d
from dataset_utils import get_dataloaders
import numpy as np


def init_model(config, dataDims, print_status=True):
    '''
    Initialize model and optimizer
    :return:
    '''
    if config.mode == 'joint modelling':
        model = FlowModel(config, dataDims)
        if config.device == 'cuda':
            model = model.cuda()

    else:
        model = CSP_model(config, dataDims)

        if config.device == 'cuda':
            model = model.cuda()

    amsgrad = False
    beta1 = config.beta1  # 0.9
    beta2 = config.beta2  # 0.999
    weight_decay = config.weight_decay  # 0.01
    momentum = 0

    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), amsgrad=amsgrad, lr=config.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), amsgrad=amsgrad, lr=config.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        print(config.optimizer + ' is not a valid optimizer')
        sys.exit()

    scheduler1 = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=15,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=15
    )
    lr_lambda = lambda epoch: 1.25
    scheduler3 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    lr_lambda2 = lambda epoch: 0.95
    scheduler4 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda2)

    nconfig = get_n_config(model)
    if print_status:
        print('Proxy model has {:.3f} million or {} parameters'.format(nconfig / 1e6, int(nconfig)))

    return model, optimizer, [scheduler1, scheduler3, scheduler4], nconfig


def get_batch_size(dataset, config):
    finished = 0
    batch_size = config.initial_batch_size.real
    batch_reduction_factor = config.auto_batch_reduction

    model, optimizer, schedulers, n_params = init_model(config, config.dataDims, print_status=False)

    while finished == 0:
        if config.device.lower() == 'cuda':
            torch.cuda.empty_cache()  # clear GPU cache

        if config.add_spherical_basis is False:  # initializing spherical basis is too expensive to do repetitively
            model, optimizer, schedulers, n_params = init_model(config, config.dataDims, print_status=False)  # for some reason necessary for memory reasons

        try:
            tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)
            if config.mode == 'joint modelling':
                flow_model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss
            elif config.mode == 'regression':
                regression_model_epoch(config, dataLoader = tr, model = model, optimizer = optimizer, update_gradients=True,iteration_override=2)
            else:
                model_epoch(config, tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)

            finished = 1

            if batch_size < 10:
                leeway = batch_reduction_factor / 2
            elif batch_size > 20:
                leeway = batch_reduction_factor
            else:
                leeway = batch_reduction_factor / 1.33

            batch_size = max(1, int(batch_size * leeway))  # give a margin for molecule sizes - larger margin for smaller batch sizes

            print('Final batch size is {}'.format(batch_size))

            tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)

            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU cache

            return tr, te, batch_size
        except:  # MemoryError or RuntimeError:
            batch_size = int(batch_size * 0.95)
            print('Training batch size reduced to {}'.format(batch_size))
            if batch_size <= 2:
                print('Model is too big! (or otherwise broken)')
                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU cache

                # for debugging purposes
                tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)
                if config.mode == 'joint modelling':
                    flow_model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss
                elif config.mode == 'regression':
                    regression_model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss
                else:
                    model_epoch(config, tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)
                sys.exit()


def model_epoch(config, dataLoader=None, model=None, optimizer=None, update_gradients=True,
                accuracy_calculation=False, iteration_override=None, log_sample_statistics=False):
    if update_gradients:
        model.train(True)
    else:
        model.eval()

    err = []
    model_pred = []
    model_targets = []
    model_probs = []
    topXsamples = []
    bottomXsamples = []
    loss_record = []
    best_losses_list = []
    worst_losses_list = []
    molecule_metrics = []
    target_wise_loss_record = []

    for i, data in enumerate(dataLoader):

        if config.device.lower() == 'cuda':
            data = data.cuda()

        assert torch.sum(torch.isnan(data.x)) == 0, "NaN in training input"

        if (not config.multi_crystal_tasks) and (not config.multi_molecule_tasks):
            losses, predictions, targets, probs = get_loss(model, data, config)
        else:
            losses, predictions, targets, probs, target_wise_loss = get_loss(model, data, config)
            target_wise_loss_record.append(target_wise_loss)

        loss = losses.mean()
        losses = losses.cpu().detach().numpy()
        loss_record.extend(losses)

        if log_sample_statistics:
            molecule_metrics.extend(data.y[2])
            topXsamples, bottomXsamples, worst_losses_list, best_losses_list = \
                get_extrema(losses, targets, probs, data,
                            worst_losses_list, best_losses_list, bottomXsamples, topXsamples,
                            num_to_keep=20)

        if accuracy_calculation:
            model_pred.extend(predictions)
            model_targets.extend(targets)
            model_probs.extend(probs)

        err.append(loss.data)  # record loss

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters

        if iteration_override is not None:
            if i >= iteration_override:
                break  # stop training early - for debugging purposes

    sample_data_dict = {}

    # logging and processing
    if log_sample_statistics:
        # get the correlation coefficient and 2d histogram of molecule features with the loss
        molecule_metrics = np.asarray(molecule_metrics)
        loss_record = np.asarray(loss_record)

        corr_coefs = {}
        loss_corr_hists = {}

        for i, key in enumerate(config.dataDims['tracking features dict']):
            corr_coefs[key + ' loss corr'] = np.corrcoef(loss_record, molecule_metrics[:, i], rowvar=False)[0, 1]
            loss_corr_hists[key + ' loss hist'] = np.histogram2d(loss_record, molecule_metrics[:, i], bins=30)

        # visualize good and bad samples
        keep_samples = 20
        worst_indices = np.argsort(worst_losses_list)[:keep_samples]
        best_indices = np.argsort(best_losses_list)[-keep_samples:]
        bottomXsamples = [bottomXsamples[worst_indices[i]] for i in range(len(worst_indices))]
        topXsamples = [topXsamples[best_indices[i]] for i in range(len(best_indices))]

        for i in range(len(bottomXsamples)):
            bottomXsamples[i][0] = draw_molecule_2d(bottomXsamples[i][-1], show=False)
        for i in range(len(bottomXsamples)):
            topXsamples[i][0] = draw_molecule_2d(topXsamples[i][-1], show=False)

        topXsamples = [sample for sample in topXsamples if sample[0] != 'failed embedding']
        bottomXsamples = [sample for sample in bottomXsamples if sample[0] != 'failed embedding']

        sample_stats = {}
        sample_stats['loss correlation coefficients'] = corr_coefs
        sample_stats['loss correlation histograms'] = loss_corr_hists
        sample_stats['best samples'] = topXsamples
        sample_stats['worst samples'] = bottomXsamples
        sample_data_dict['sample stats'] = sample_stats

    if accuracy_calculation:
        metrics = getAccuracy(config, model_pred, model_targets, model_probs)
        sample_data_dict['accuracy metrics'] = metrics
        sample_data_dict['model probabilities'] = np.asarray(model_probs)
        sample_data_dict['sample targets'] = np.asarray(model_targets)
        sample_data_dict['model predictions'] = np.asarray(model_pred)

    if accuracy_calculation or log_sample_statistics:
        return err, np.asarray(loss_record), sample_data_dict, np.asarray(target_wise_loss_record)
    else:
        return err, np.asarray(loss_record), np.asarray(target_wise_loss_record)


def flow_model_epoch(config, dataLoader=None, model=None, optimizer=None,
                     update_gradients=True, iteration_override=None, ):
    if update_gradients:
        model.train(True)
    else:
        model.eval()

    err = []
    loss_record = []

    for i, data in enumerate(dataLoader):
        if config.device.lower() == 'cuda':
            data = data.cuda()

        losses = get_flow_loss(model, data)
        loss = losses.mean()
        err.append(loss.data.cpu())  # record loss
        loss_record.extend(losses.cpu().detach().numpy())

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation

            # record gradients and print statistics
            # grads = []
            # for p in list(filter(lambda p: p.grad is not None, model.parameters())):
            #     grads.append(p.grad.data.norm(2).item())
            #
            # print('Average = {:.3f} Max = {:.3f}'.format(np.average(grads),np.amax(grads)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # grads = []
            # for p in list(filter(lambda p: p.grad is not None, model.parameters())):
            #     grads.append(p.grad.data.norm(2).item())
            #
            # print('Post-clip Average = {:.3f} Max = {:.3f}'.format(np.average(grads),np.amax(grads)))

            optimizer.step()  # update parameters

        if iteration_override is not None:
            if i >= iteration_override:
                break  # stop training early - for debugging purposes

    return err, loss_record


def regression_model_epoch(config, dataLoader=None, model=None, optimizer=None,
                           update_gradients=True, iteration_override=None, record_stats = False):
    if update_gradients:
        model.train(True)
    else:
        model.eval()

    err = []
    loss_record = []
    epoch_stats_dict = {
        'prediction': [],
        'target' : [],
        'tracking features': [],
    }

    for i, data in enumerate(dataLoader):
        if record_stats:
            epoch_stats_dict['target'].extend(data.y[0])
            epoch_stats_dict['tracking features'].extend(data.y[2])

        if config.device.lower() == 'cuda':
            data = data.cuda()

        losses, pred, target = get_regression_loss(model, data)
        loss = losses.mean()
        err.append(loss.data.cpu())  # record loss
        loss_record.extend(losses.cpu().detach().numpy())
        if record_stats:
            epoch_stats_dict['prediction'].extend(pred)


        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters

        if iteration_override is not None:
            if i >= iteration_override:
                break  # stop training early - for debugging purposes
    if record_stats:
        return err, loss_record, epoch_stats_dict
    else:
        return err, loss_record


def get_grad_norm(model):
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(params) == 0:
        norm = 0
    else:
        norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).cpu() for p in params]), 2.0).item()

    return norm


def set_lr(schedulers, optimizer, config, err_tr, hit_max_lr):
    if config.lr_schedule:
        lr = optimizer.param_groups[0]['lr']
        if lr > 1e-4:
            schedulers[0].step(torch.mean(torch.stack(err_tr)))  # plateau scheduler

        if not hit_max_lr:
            if lr <= config.max_lr:
                schedulers[1].step()
            else:
                hit_max_lr = True
        elif hit_max_lr:
            if lr > config.learning_rate:
                schedulers[2].step()  # start reducing lr
    lr = optimizer.param_groups[0]['lr']
    print("Learning rate is {:.5f}".format(lr))
    return optimizer


def getNaiveTestLoss(data, config):
    output = torch.Tensor(config.classData['class probs'])
    output = output.repeat(len(data.y[0]), 1)

    if data.y[0].ndim > 1:
        loss = F.cross_entropy(output, data.y[0][:, 0].long().cpu())
    else:
        loss = F.cross_entropy(output, data.y[0].long().cpu())

    return loss


def get_flow_loss(model, data):
    zs, prior_logprob, log_det = model(data)
    logprob = prior_logprob + log_det

    return -(logprob)


def get_regression_loss(model, data):
    pred = model(data)
    targets = data.y[0]

    if targets.ndim > 1:
        targets = targets[:, 0]

    if pred.ndim > 1:
        pred = pred[:, 0]

    losses = F.smooth_l1_loss(pred, targets.float(),reduction='none')

    return losses, pred.cpu().detach().numpy(), targets.cpu().detach().numpy()


def get_loss(model, data, config):
    """
    get the regression loss on a batch of datapoints
    :param train_data: sequences and scores
    :return: model loss over the batch
    """
    output = model(data)  # reshape output from flat filters to channels * filters per channel

    if (not config.multi_crystal_tasks) and (not config.multi_molecule_tasks):  # single categorical task
        targets = data.y[0]

        if targets.ndim > 1:
            targets = targets[:, 0]

        losses = F.cross_entropy(output, targets.long(), reduction='none')
        probs = F.softmax(output, dim=1).cpu().detach().numpy()
        predictions = torch.argmax(output, dim=1)

        return losses, predictions.cpu().detach().numpy(), targets.cpu().detach().numpy(), probs

    else:
        targets = data.y[0][:, 0]
        loss_i = []
        for i, key in enumerate(config.dataDims['target features dict']['target feature keys']):
            dtype = config.dataDims['target features dict'][key]['dtype']
            if dtype in ('int', 'float'):
                loss_i.append(F.smooth_l1_loss(output[i][:, 0], data.y[0][:, i].float(), reduction='none'))
            elif dtype in ('bool', 'str'):
                loss_i.append(F.cross_entropy(output[i], data.y[0][:, i].long(), reduction='none'))
            if key == config.target:
                loss_i[-1] *= 5

        target_wise_loss = torch.mean(torch.stack(loss_i), dim=1).cpu().detach().numpy()
        losses = torch.mean(torch.stack(loss_i), dim=0).to(data.y[0].device)  # mean over loss functions, but not over samples

        probs = F.softmax(output[0], dim=1).cpu().detach().numpy()
        predictions = torch.argmax(output[0], dim=1)

        return losses, predictions.cpu().detach().numpy(), targets.cpu().detach().numpy(), probs, target_wise_loss


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


def getAccuracy(config, predictions, targets, probs):
    probs = np.asarray(probs)
    targets = np.asarray(targets).astype(int)

    overallTop1Accuracy, byGroupTop1Accuracy = computeTopXAccuracy(config, probs, targets, X=1)
    X = min(config.dataDims['output classes'][0] // 2, 5)
    overallTopXAccuracy, byGroupTopXAccuracy = computeTopXAccuracy(config, probs, targets, X=X)

    predictions = np.asarray(predictions)
    nClasses = config.dataDims['output classes'][0]

    if targets.ndim > 1:
        targets = targets[:, 0]

    prob_matrix = np.zeros((nClasses, nClasses))
    target_inds = [np.where(targets == i) for i in range(nClasses)]
    for i in range(nClasses):
        prob_matrix[i, :] = np.sum(probs[target_inds[i]], axis=0)

    confusion_matrix = metrics.confusion_matrix(targets, predictions)

    avgProbAccuracy, avgProbPrecision, avgProbRecall, avgProbF1 = computeF1(prob_matrix.astype(int), nClasses)
    avgAccuracy, avgPrecision, avgRecall, avgF1 = computeF1(confusion_matrix, nClasses)

    if probs.shape[1] == 2:
        roc_score = metrics.roc_auc_score(y_true=targets, y_score=probs[:, 1], average='macro', multi_class='ovo')
    else:
        roc_score = metrics.roc_auc_score(y_true=targets, y_score=probs, average='macro', multi_class='ovo')

    quasi_regression_score = metrics.mean_squared_error(y_true=targets / nClasses, y_pred=predictions / nClasses)

    print("Probability Matrix:")
    normed_prob_mat = prob_matrix / np.sum(prob_matrix)
    print('{}'.format((normed_prob_mat * 100 * 100).astype(int)))
    print('Prob based Accuracy {:.3f} Precision {:.3f} Recall {:.3f} F1 {:.3f}'.format(avgProbAccuracy, avgProbPrecision, avgProbRecall, avgProbF1))
    print("Confusion Matrix:")
    print('{}'.format(confusion_matrix))
    print('Accuracy {:.3f} Precision {:.3f} Recall {:.3f} F1 {:.3f}'.format(avgAccuracy, avgPrecision, avgRecall, avgF1))
    print('Top 1 Accuracy Overall: {:.3f} By Group: {:.3f}'.format(overallTop1Accuracy, np.average(byGroupTop1Accuracy)))
    print('Top {} Accuracy Overall: {:.3f} By Group: {:.3f}'.format(X, overallTopXAccuracy, np.average(byGroupTopXAccuracy)))
    print('ROC AUC Score {:.3f}'.format(roc_score))

    metrics_dict = {
        'ROC AUC': roc_score,
        'quasi L2': quasi_regression_score,
        'F1': avgF1,
        'P F1': avgProbF1,
        'precision': avgPrecision,
        'P precision': avgProbPrecision,
        'recall': avgRecall,
        'P recall': avgProbRecall,
        'accuracy': avgAccuracy,
        'P accuracy': avgProbAccuracy,
        'confusion matrix': confusion_matrix,
        'P confusion matrix': prob_matrix,
        'average top 1 accuracy': overallTop1Accuracy,
        'average top {} accuracy'.format(X): overallTopXAccuracy
    }
    if 'symbol' in config.target:
        for i in range(len(config.groupLabels)):
            metrics_dict[config.groupLabels[i] + ' top 1 accuracy'] = byGroupTop1Accuracy[i]
            metrics_dict[config.groupLabels[i] + ' top {} accuracy'.format(X)] = byGroupTopXAccuracy[i]
    else:
        for i in range(len(byGroupTop1Accuracy)):
            metrics_dict['Class {} top 1 accuracy'.format(i)] = byGroupTop1Accuracy[i]
            metrics_dict['Class {} top {} accuracy'.format(i, X)] = byGroupTopXAccuracy[i]

    return metrics_dict


def checkConvergence(config, record):
    """
    check if we are converged
    condition: test loss has increased or levelled out over the last several epochs
    :return: convergence flag
    """

    converged = False
    if type(record) == list:
        record = np.asarray(record)

    if len(record) > (config.history + 2):
        if all(record[-config.history:] > np.amin(record)):
            converged = True
            print("Model converged, target diverging")

        criteria = np.var(record[-config.history:]) / np.abs(np.average(record[-config.history:]))
        print('Convergence criteria at {:.3f}'.format(np.log10(criteria)))
        if criteria < config.convergence_eps:
            converged = True
            print("Model converged, target stabilized")

    return converged


def get_extrema(losses, targets, probs, data, worst_losses_list, best_losses_list, bottomXsamples, topXsamples, num_to_keep=1):
    X = max(min(num_to_keep, len(losses) // 5), 1)
    worst_indices = np.argsort(losses)[-X:]
    best_indices = np.argsort(losses)[:X]
    worst_losses = losses[worst_indices]
    best_losses = losses[best_indices]
    worst_losses_list.extend(worst_losses)
    best_losses_list.extend(best_losses)

    correct_answers = targets[worst_indices]
    guesses = np.argmax(probs[worst_indices], axis=1)

    for idx in range(len(worst_indices)):
        bottomXsamples.append([worst_losses[idx], correct_answers[idx], guesses[idx], data.y[1][worst_indices[idx]]])

    correct_answers = targets[best_indices]
    guesses = np.argmax(probs[best_indices], axis=1)

    for idx in range(len(best_indices)):
        topXsamples.append([best_losses[idx], correct_answers[idx], guesses[idx], data.y[1][best_indices[idx]]])

    return bottomXsamples, topXsamples, worst_losses_list, best_losses_list


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
