
def log_accuracy(self, epoch, dataset_builder, train_loader, test_loader,
                 te_record, epoch_stats_dict, config, model, wandb_log_figures=True):
    t0 = time.time()

    # correlate losses with molecular features
    tracking_features = np.asarray(epoch_stats_dict['tracking features'])
    loss_correlations = np.zeros(config.dataDims['n tracking features'])
    features = []
    for i in range(config.dataDims['n tracking features']):
        features.append(config.dataDims['tracking features dict'][i])
        loss_correlations[i] = np.corrcoef(te_record, tracking_features[:, i], rowvar=False)[0, 1]

    sort_inds = np.argsort(loss_correlations)
    loss_correlations = loss_correlations[sort_inds]

    if wandb_log_figures:
        fig = go.Figure(go.Bar(
            y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['n tracking features'])],
            x=[loss_correlations[i] for i in range(config.dataDims['n tracking features'])],
            orientation='h',
        ))
        wandb.log({'Loss correlations': fig})

    if 'regression' in config.mode:
        target_mean = config.dataDims['mean']
        target_std = config.dataDims['std']

        target = np.asarray(epoch_stats_dict['targets'])
        prediction = np.asarray(epoch_stats_dict['predictions'])
        orig_target = target * target_std + target_mean
        orig_prediction = prediction * target_std + target_mean

        losses = ['normed error', 'abs normed error', 'squared error']
        loss_dict = {}
        losses_dict = {}
        for loss in losses:
            if loss == 'normed error':
                loss_i = (orig_target - orig_prediction) / np.abs(orig_target)
            elif loss == 'abs normed error':
                loss_i = np.abs((orig_target - orig_prediction) / np.abs(orig_target))
            elif loss == 'squared error':
                loss_i = (orig_target - orig_prediction) ** 2
            losses_dict[loss] = loss_i  # huge unnecessary upload
            loss_dict[loss + ' mean'] = np.mean(loss_i)
            loss_dict[loss + ' std'] = np.std(loss_i)
            print(loss + ' mean: {:.3f} std: {:.3f}'.format(loss_dict[loss + ' mean'], loss_dict[loss + ' std']))

        wandb.log(loss_dict)

        # log loss distribution
        if wandb_log_figures:
            fig = go.Figure()
            for loss in losses:
                fig.add_trace(go.Histogram(
                    x=losses_dict[loss],
                    histnorm='probability density',
                    nbinsx=100,
                    name=loss,
                    showlegend=True,
                    opacity=0.55
                ))
            fig.update_layout(barmode='overlay')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            wandb.log({'loss histograms': fig})

            # log target distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=orig_target,
                histnorm='probability density',
                nbinsx=100,
                name='target',
                showlegend=True,
                opacity=1
            ))
            fig.add_trace(go.Histogram(
                x=orig_prediction,
                histnorm='probability density',
                nbinsx=100,
                name='prediction',
                showlegend=True,
                opacity=0.65
            ))
            fig.update_layout(barmode='overlay')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            wandb.log({'target distribution': fig})

            # predictions vs target trace
            xline = np.linspace(min(min(orig_target), min(orig_prediction)), max(max(orig_target), max(orig_prediction)), 1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=orig_target, y=orig_prediction, mode='markers', showlegend=False))
            fig.add_trace(go.Scatter(x=xline, y=xline, showlegend=False))
            fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
            wandb.log({'Prediction Trace': fig})

    elif 'classification' in config.mode:
        probs = np.asarray(epoch_stats_dict['predictions'])
        targets = np.asarray(epoch_stats_dict['targets']).astype(int)
        predictions = np.argmax(probs, axis=1)
        nClasses = config.dataDims['output classes'][0]

        # get classwise accuracy
        overallTop1Accuracy, byGroupTop1Accuracy = computeTopXAccuracy(config, probs, targets, X=1)
        X = min(nClasses // 2, 5)
        overallTopXAccuracy, byGroupTopXAccuracy = computeTopXAccuracy(config, probs, targets, X=X)

        if targets.ndim > 1:
            targets = targets[:, 0]

        # get confusion matrix
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

        # report scores
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

        classifier_accuracy_dict = {
            'ROC AUC': roc_score,
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
        wandb.log(classifier_accuracy_dict)
        accuracy_dict = {}
        for i in range(len(self.class_labels)):
            accuracy_dict[self.class_labels[i] + ' top 1 accuracy'] = byGroupTop1Accuracy[i]
            accuracy_dict[self.class_labels[i] + ' top {} accuracy'.format(X)] = byGroupTopXAccuracy[i]

        if wandb_log_figures:
            xaxis = self.class_labels
            yaxis = self.class_labels[-1::-1]
            zaxis = confusion_matrix / np.sum(confusion_matrix)
            fig = go.Figure(data=go.Heatmap(z=np.flipud(zaxis),
                                            x=xaxis,
                                            y=yaxis,
                                            zmin=0,
                                            # zmax=1,
                                            ))
            wandb.log({"Confusion Matrix": fig})

            xaxis = self.class_labels
            yaxis = self.class_labels[-1::-1]
            zaxis = prob_matrix / np.sum(prob_matrix)
            fig = go.Figure(data=go.Heatmap(z=np.flipud(zaxis),
                                            x=xaxis,
                                            y=yaxis,
                                            zmin=0,
                                            # zmax=1,
                                            ))
            wandb.log({"P Confusion Matrix": fig})

            rands = np.random.choice(len(targets), size=len(targets), replace=False)  # min(len(targets), 9999)
            wandb.log({"roc {}".format(epoch): wandb.plot.roc_curve(
                y_true=targets[rands], y_probas=probs[rands], labels=self.class_labels, title='Epoch {}'.format(epoch))})

    elif config.mode == 'joint modelling':
        '''
        Get the samples
        '''
        n_samples = config.num_samples
        n_dims = self.dataDims['n crystal features']
        dataset_length = len(test_loader.dataset)
        self.sampling_batch_size = min(dataset_length, config.final_batch_size)
        n_repeats = max(n_samples // dataset_length, 1)
        n_samples = n_repeats * dataset_length
        model.eval()

        if config.device.lower() == 'cuda':
            torch.cuda.empty_cache()  # clear GPU

        # boilerplate
        targets, train_data = self.get_generation_conditions(train_loader, test_loader, model, config)
        self.check_inversion_quality(model, test_loader, config)
        pca = model.fit_pca(train_data, print_variance=(epoch == 0))
        # get all our samples
        sample_dict = {}
        sample_dict['data'] = targets
        sample_dict['independent gaussian'] = model.prior.sample((n_samples,)).detach().numpy()
        sample_dict['pca gaussian'] = model.pca_sampling(pca, n_samples)
        sample_dict['nf gaussian'] = self.sample_nf(n_repeats, config, model, test_loader)
        renormalized_sample_dict = {}
        for key in sample_dict.keys():
            renormalized_sample_dict[key] = model.destandardize_samples(sample_dict[key], self.dataDims)

        # skipping because it's expensive
        # # get PC and NF scores
        # pc_scores_dict = self.get_pc_scores(sample_dict, pca)
        # nf_scores_dict = self.get_nf_scores(sample_dict, model, config, test_loader, n_repeats, dataset_length)
        #
        # wandb.log({
        #     'data nf score': np.average(nf_scores_dict['data']),
        #     'pca gaussian nf score': np.average(nf_scores_dict['pca gaussian']),
        #     'independent gaussian nf score': np.average(nf_scores_dict['independent gaussian']),
        #     'nf gaussian nf score': np.average(nf_scores_dict['nf gaussian']),
        #     'data pc score': np.average(pc_scores_dict['data']),
        #     'pca gaussian pc score': np.average(pc_scores_dict['pca gaussian']),
        #     'independent gaussian pc score': np.average(pc_scores_dict['independent gaussian']),
        #     'nf gaussian pc score': np.average(pc_scores_dict['nf gaussian']),
        # })

        # check sample efficiency
        print("Computing sample efficiency")
        sample_efficiency_dict = {}
        feature_accuracy_dict = {}
        for i, (key, sample) in enumerate(renormalized_sample_dict.items()):
            if sample.ndim == 2:
                if sample.shape[0] == dataset_length * n_repeats:
                    sample = sample.reshape(dataset_length, n_repeats, n_dims)  # roll up the first dim for the indepenent and pc sampels
                elif sample.shape[0] == dataset_length:
                    sample = sample[:, None, :]  # the real data only has one repeat

            sample_efficiency_dict, feature_accuracy_dict = self.get_sample_efficiency(self.dataDims, targets, sample, sample_efficiency_dict, feature_accuracy_dict, key)
            print(key + ' average error {:.3f}'.format(sample_efficiency_dict[key + ' average mae']))

        wandb.log(sample_efficiency_dict)
        '''
        compute and report some overlaps
        '''
        # flatten sample dimension for the nf data for subsequent analysis
        sample_shape = renormalized_sample_dict['nf gaussian'].shape
        sample_dict['nf gaussian'] = sample_dict['nf gaussian'].reshape(sample_shape[0] * sample_shape[1], sample_shape[2])
        renormalized_sample_dict['nf gaussian'] = renormalized_sample_dict['nf gaussian'].reshape(sample_shape[0] * sample_shape[1], sample_shape[2])

        # 1D histogram overlaps
        overlaps_1d = {}
        for j in range(n_dims):
            mini, maxi = np.amin(renormalized_sample_dict['data'][:, j]), np.amax(renormalized_sample_dict['data'][:, j])
            h1, r1 = np.histogram(renormalized_sample_dict['data'][:, j], bins=100, range=(mini, maxi))
            h1 = h1 / len(renormalized_sample_dict['data'][:, j])
            for i, key in enumerate(['independent gaussian', 'pca gaussian', 'nf gaussian']):
                h2, r2 = np.histogram(renormalized_sample_dict[key][:, j], bins=r1)
                h2 = h2 / len(renormalized_sample_dict[key][:, j])
                overlaps_1d[key + ' ' + self.dataDims['crystal features'][j]] = np.min(np.concatenate((h1[None], h2[None]), axis=0), axis=0).sum()

        average_independent_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys() if 'independent' in key])
        average_pc_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys() if 'pc' in key])
        average_nf_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys() if 'nf' in key])

        print("1D Overlaps With Data: Ind. {:.3f} PC {:.3f} NF {:.3f}".format(average_independent_overlap, average_pc_overlap, average_nf_overlap))
        wandb.log({
            'independent 1D overlap': average_independent_overlap,
            'pc 1D overlap': average_pc_overlap,
            'nf 1D overlap': average_nf_overlap
        })

        # 2D histogram overlaps
        overlaps_2d = {}
        for i in range(n_dims):
            minx, maxx = np.amin(renormalized_sample_dict['data'][:, i]), np.amax(renormalized_sample_dict['data'][:, i])

            for j in range(n_dims):
                if i != j:
                    miny, maxy = np.amin(renormalized_sample_dict['data'][:, j]), np.amax(renormalized_sample_dict['data'][:, j])
                    h1, x1, y1 = np.histogram2d(renormalized_sample_dict['data'][:, i], renormalized_sample_dict['data'][:, j], bins=100, range=((minx, maxx), (miny, maxy)))
                    h1 = h1 / len(renormalized_sample_dict['data'][:, j])
                    for key in ['independent gaussian', 'pca gaussian', 'nf gaussian']:
                        h2, x2, y2 = np.histogram2d(renormalized_sample_dict[key][:, i], renormalized_sample_dict[key][:, j], bins=(x1, y1))
                        h2 = h2 / len(renormalized_sample_dict[key][:, j])
                        overlaps_2d[key + ' ' + self.dataDims['crystal features'][i] + ' vs ' + self.dataDims['crystal features'][j]] = np.min(np.concatenate((h1.flatten()[None], h2.flatten()[None]), axis=0), axis=0).sum()

        average_independent_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'independent' in key])
        average_pc_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'pc' in key])
        average_nf_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'nf' in key])

        print("2D Overlaps With Data: Ind. {:.3f} PC {:.3f} NF {:.3f}".format(average_independent_overlap, average_pc_overlap, average_nf_overlap))
        wandb.log({
            'independent 2D overlap': average_independent_overlap,
            'pc 2D overlap': average_pc_overlap,
            'nf 2D overlap': average_nf_overlap
        })

        if wandb_log_figures:
            fig_dict = {}

            # bar graph of feature-wise sample accuracy
            feat_keys = list(feature_accuracy_dict.keys())
            for key in feat_keys:
                if 'data' in key:
                    feature_accuracy_dict.pop(key)
            feat_keys = list(feature_accuracy_dict.keys())
            indy_list = [key for key in feat_keys if 'independent gaussian' in key]
            pc_list = [key for key in feat_keys if 'pca gaussian' in key]
            nf_list = [key for key in feat_keys if 'nf gaussian' in key]
            feat_keys = [val for triplet in zip(*[indy_list, pc_list, nf_list]) for val in triplet]
            color = []
            for key in [key for key in feat_keys if 'mae' in key]:
                if 'independent' in key:
                    color.append('red')
                elif 'pc' in key:
                    color.append('blue')
                elif 'nf' in key:
                    color.append('green')
            fig = go.Figure(go.Bar(
                y=[key for key in feat_keys if 'mae' in key],
                x=[feature_accuracy_dict[key] for key in feat_keys if 'mae' in key],
                orientation='h',
                marker=dict(color=color)
            ))
            fig_dict['Feature-wise MAE'] = fig
            color = []
            for key in [key for key in feat_keys if '0.05' in key]:
                if 'independent' in key:
                    color.append('red')
                elif 'pc' in key:
                    color.append('blue')
                elif 'nf' in key:
                    color.append('green')
            fig = go.Figure(go.Bar(
                y=[key for key in feat_keys if '0.05' in key],
                x=[feature_accuracy_dict[key] for key in feat_keys if '0.05' in key],
                orientation='h',
                marker=dict(color=color)
            ))
            fig_dict['Feature-wise 0.05 efficiency'] = fig

            # bar graph of 1d overlaps
            color = []
            for key in overlaps_1d.keys():
                if 'independent' in key:
                    color.append('red')
                elif 'pc' in key:
                    color.append('blue')
                elif 'nf' in key:
                    color.append('green')
            fig = go.Figure(go.Bar(
                y=list(overlaps_1d.keys()),
                x=[overlaps_1d[key] for key in overlaps_1d],
                orientation='h',
                marker=dict(color=color)
            ))
            fig_dict['1D overlaps'] = fig

            color = []
            for key in overlaps_2d.keys():
                if 'independent' in key:
                    color.append('red')
                elif 'pc' in key:
                    color.append('blue')
                elif 'nf' in key:
                    color.append('green')
            fig = go.Figure(go.Bar(
                y=list(overlaps_2d.keys()),
                x=[overlaps_2d[key] for key in overlaps_2d],
                orientation='h',
                marker=dict(color=color)
            ))
            fig_dict['2D overlaps'] = fig

            # 1d Histograms
            for i in range(n_dims):
                fig = go.Figure()
                for key in renormalized_sample_dict.keys():
                    fig.add_trace(go.Histogram(
                        x=renormalized_sample_dict[key][:, i],
                        histnorm='probability density',
                        nbinsx=100,
                        name=key,
                        showlegend=True,
                    ))
                fig.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig.update_traces(opacity=0.5)

                fig_dict[self.dataDims['crystal features'][i] + ' distribution'] = fig

            # 2d Scatterplots -
            pairs = []  # pick some dims randomly
            np.random.seed(1)
            for i in range(min(self.n_crystal_dims, self.n_crystal_dims ** 2 // 2)):
                pairs.append(np.random.choice(np.arange(self.n_crystal_dims), size=2, replace=False))
            colors = 'black', 'green', 'orange', 'red'
            nbins = 50
            for n in range(len(pairs)):
                i, j = pairs[n]
                fig = go.Figure()
                for k, key in enumerate(['data', 'independent gaussian', 'pca gaussian', 'nf gaussian']):  # renormalized_sample_dict.keys()):
                    if key == 'data':
                        opacity = 1
                    else:
                        opacity = 0.75
                    fig.add_trace(go.Histogram2dContour(
                        x=renormalized_sample_dict[key][:, i],
                        y=renormalized_sample_dict[key][:, j],
                        histnorm='probability density',
                        name=key,
                        showlegend=True,
                        nbinsx=nbins,
                        nbinsy=nbins,
                        contours=dict(coloring="none"),
                        line_color=colors[k],
                        line_width=0.5,
                        ncontours=50,
                        opacity=opacity
                    ))

                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', )
                fig_dict[self.dataDims['crystal features'][i] + ' vs ' + self.dataDims['crystal features'][j]] = fig
                #
                # # Score Histograms
                # for n in range(2):
                #     fig = go.Figure()
                #     for i, key in enumerate(pc_scores_dict.keys()):
                #         if n == 0:
                #             values = pc_scores_dict[key]
                #
                #         elif n == 1:
                #             values = nf_scores_dict[key]
                #
                #         if i == 0:
                #             opacity = 1
                #         else:
                #             opacity = 0.5
                #
                #         if i == 0:
                #             xstart = np.quantile(values, 0.01)
                #             xend = np.quantile(values, 0.99)
                #         values = np.clip(values, a_min=xstart, a_max=xend)
                #         fig.add_trace(go.Histogram(
                #             x=values,
                #             histnorm='probability density',
                #             nbinsx=100,
                #             name=key,
                #             showlegend=True,
                #             opacity=opacity,
                #         ))
                #         fig.update_layout(barmode='overlay')
                #     fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                #     # fig.show()
                #     if n == 0:
                #         fig_dict['PCA log-P'] = fig
                #     elif n == 1:
                #         fig_dict['NF log-P'] = fig

                keys = list(fig_dict.keys())
                for key in keys:
                    if '/' in key:
                        new_key = key.replace('/', '')
                        fig_dict[new_key] = fig_dict.pop(key)
                wandb.log(fig_dict)

    else:
        print(config.mode + ' is not a real model! how did you get this far?')

    print('Analysis took {} seconds'.format(int(time.time() - t0)))


    def model_epoch(self, config, dataLoader=None, model=None, optimizer=None, update_gradients=True,
                    iteration_override=None, record_stats=False):
        t0 = time.time()
        if update_gradients:
            model.train(True)
        else:
            model.eval()

        err = []
        loss_record = []
        epoch_stats_dict = {
            'predictions': [],
            'targets': [],
            'tracking features': [],
        }

        for i, data in enumerate(dataLoader):
            if 'cell' in config.mode:
                t0 = time.time()
                real_sample = np.random.uniform(0, 1, size=data.num_graphs) < config.csd_fraction
                data = self.supercell_data(real_sample, data, config)
                if i < 3:
                    print('Batch {} supercell generation took {:.2f} seconds for {} samples'.format(i, round(time.time() - t0, 2), data.num_graphs))

            if config.device.lower() == 'cuda':
                data = data.cuda()

            if config.test_mode or config.anomaly_detection:
                assert torch.sum(torch.isnan(data.x)) == 0, "NaN in training input"

            losses, predictions = self.get_loss(model, data, config)

            loss = losses.mean()
            err.append(loss.data.cpu())  # average loss
            loss_record.extend(losses.cpu().detach().numpy())  # loss distribution

            if record_stats:
                epoch_stats_dict['predictions'].extend(predictions)
                epoch_stats_dict['targets'].extend(data.y[0].cpu().detach().numpy())
                epoch_stats_dict['tracking features'].extend(data.y[2])

            if update_gradients:
                optimizer.zero_grad()  # reset gradients from previous passes
                loss.backward()  # back-propagation
                optimizer.step()  # update parameters

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0
        if record_stats:
            return err, loss_record, epoch_stats_dict, total_time
        else:
            return err, loss_record, total_time

    def get_loss(self, model, data, config):
        if (config.mode == 'single molecule regression') or (config.mode == 'cell regression'):
            pred = model(data)
            targets = data.y[0]
            if targets.ndim > 1:
                targets = targets[:, 0]

            if pred.ndim > 1:
                pred = pred[:, 0]

            losses = F.smooth_l1_loss(pred, targets.float(), reduction='none')
            return losses, pred.cpu().detach().numpy()

        elif (config.mode == 'single molecule classification') or (config.mode == 'cell classification'):
            output = model(data)  # reshape output from flat filters to channels * filters per channel
            targets = data.y[0]

            if targets.ndim > 1:
                targets = targets[:, 0]

            losses = F.cross_entropy(output, targets.long(), reduction='none')
            probs = F.softmax(output, dim=1).cpu().detach().numpy()

            return losses, probs

        elif config.mode == 'joint modelling':
            zs, prior_logprob, log_det = model(data)
            logprob = prior_logprob + log_det

            return -(logprob), prior_logprob.cpu().detach().numpy()