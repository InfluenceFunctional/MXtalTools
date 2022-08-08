
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



    def get_batch_size(self, dataset, config):
        finished = 0
        batch_size = config.max_batch_size.real
        batch_reduction_factor = config.auto_batch_reduction

        if 'gan'.lower() in config.mode:
            generator, discriminator, g_optimizer, \
            g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)

            while finished == 0:
                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU cache
                    generator.cuda()
                    discriminator.cuda()

                if config.add_spherical_basis is False:  # initializing spherical basis is too expensive to do repetitively
                    generator, discriminator, g_optimizer, \
                    g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)

                try:
                    train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=batch_size)
                    d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                        self.gan_epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                       g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                       update_gradients=True, record_stats=True, iteration_override=2)  # train & compute test loss

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
                        train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=batch_size)
                        d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                            self.gan_epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                           g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                           update_gradients=True, record_stats=True, iteration_override=2)  # train & compute test loss

                        sys.exit()

        else:

            model, optimizer, schedulers, n_params = self.init_model(config, config.dataDims, print_status=False)

            while finished == 0:
                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU cache

                if config.add_spherical_basis is False:  # initializing spherical basis is too expensive to do repetitively
                    model, optimizer, schedulers, n_params = self.init_model(config, config.dataDims, print_status=False)  # for some reason necessary for memory reasons

                try:
                    tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)
                    self.model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss

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
                        self.model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss

                        sys.exit()



    def generated_supercells(self, data, config, generator):
        '''
              test code for on-the-fly cell generation
              data = self.build_supercells(data)
              0. extract molecule and cell parameters
              1. find centroid
              2. find principal axis & angular component
              3. place centroid & align axes
              4. apply point symmetry
              5. tile supercell
              '''
        sg_numbers = [int(data.y[2][i][self.sg_number_ind]) for i in range(data.num_graphs)]
        lattices = [self.lattice_type[number] for number in sg_numbers]

        cell_sample = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device)).cpu()
        cell_lengths, cell_angles, rand_position, rand_rotation = cell_sample.split(3, 1)
        cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
            cell_lengths, cell_angles, rand_position, rand_rotation, lattices, config.dataDims)

        for i in range(data.num_graphs):
            # 0 extract molecule and cell parameters
            atoms = np.asarray(data.x[data.batch == i].cpu().detach())
            atomic_numbers = np.asarray(atoms[:, 0])
            heavy_inds = np.where(atomic_numbers != 1)
            atoms = atoms[heavy_inds]

            # get symmetry info
            sym_ops = np.asarray(self.sym_ops[sg_numbers[i]])  # symmetry operations between the general positions for this space group
            z_value = len(sym_ops)  # number of molecules in the reference cell

            # prep conformer
            coords = np.asarray(data.pos[data.batch == i].cpu().detach())
            weights = np.asarray([self.atom_weights[int(number)] for number in atomic_numbers])
            coords = coords[heavy_inds]
            weights = weights[heavy_inds]

            # # option to get the cell itself from the CSD
            # cell_lengths = data.y[2][i][self.cell_length_inds]  # pull cell params from tracking inds
            # cell_angles = data.y[2][i][self.cell_angle_inds]

            T_fc = coor_trans_matrix('f_to_c', cell_lengths[i].detach().numpy(), cell_angles[i].detach().numpy())
            T_cf = coor_trans_matrix('c_to_f', cell_lengths[i].detach().numpy(), cell_angles[i].detach().numpy())
            cell_vectors = T_fc.dot(np.eye(3)).T

            random_coords = randomize_molecule_position_and_orientation(
                coords.astype(float), weights.astype(float), T_fc.astype(float), T_cf.astype(float),
                np.asarray(self.sym_ops[sg_numbers[i]], dtype=float), set_position=rand_position[i].detach().numpy(), set_rotation=rand_rotation[i].detach().numpy())

            reference_cell, ref_cell_f = build_random_crystal(T_cf, T_fc, random_coords, np.asarray(self.sym_ops[sg_numbers[i]], dtype=float), z_value)

            supercell_atoms, supercell_coords = ref_to_supercell(reference_cell, z_value, atoms, cell_vectors)

            supercell_batch = torch.ones(len(supercell_atoms)).int() * i

            # append supercell info to the data class
            if i == 0:
                new_x = supercell_atoms
                new_coords = supercell_coords
                new_batch = supercell_batch
                new_ptr = torch.zeros(data.num_graphs)
            else:
                new_x = torch.cat((new_x, supercell_atoms), dim=0)
                new_coords = torch.cat((new_coords, supercell_coords), dim=0)
                new_batch = torch.cat((new_batch, supercell_batch))
                new_ptr[i] = new_ptr[-1] + len(new_x)

        # update dataloader with cell info
        data.x = new_x.type(dtype=torch.float32)
        data.pos = new_coords.type(dtype=torch.float32)
        data.batch = new_batch.type(dtype=torch.int64)
        data.ptr = new_ptr.type(dtype=torch.int64)

        return data, cell_sample



    def train(self):
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=self.config.wandb.experiment_tag):
            # config = wandb.config # todo: wandb configs don't support nested namespaces. Sweeps are officially broken
            # print(config)

            config = self.config  # go with hand-built config

            # dataset
            dataset_builder = BuildDataset(config, self.point_groups, premade_dataset=self.prep_dataset)
            del self.prep_dataset  # we don't actually want this huge thing floating around

            config.dataDims = dataset_builder.get_dimension()
            self.dataDims = dataset_builder.get_dimension()
            if 'classification' in config.mode:  # for convenience
                self.class_labels = self.dataDims['class labels']
                self.class_weights = self.dataDims['class weights']
            if config.mode == 'joint modelling':
                self.lattice_features = dataset_builder.lattice_keys
                self.n_crystal_dims = self.dataDims['n crystal features']
                if config.generator.conditional_modelling:
                    self.n_conditional_features = self.dataDims['n conditional features']
                else:
                    self.n_conditional_features = 0
            if 'cell' in config.mode:  # get relevant indices
                self.cell_angle_keys = ['crystal alpha', 'crystal beta', 'crystal gamma']
                self.cell_angle_inds = [self.dataDims['tracking features dict'].index(key) for key in self.cell_angle_keys]
                self.cell_length_keys = ['crystal cell a', 'crystal cell b', 'crystal cell c']
                self.cell_length_inds = [self.dataDims['tracking features dict'].index(key) for key in self.cell_length_keys]
                self.z_value_ind = self.dataDims['tracking features dict'].index('crystal z value')
                self.sg_number_ind = self.dataDims['tracking features dict'].index('crystal spacegroup number')
                self.mol_volume_ind = self.dataDims['tracking features dict'].index('molecule volume')
                self.crystal_packing_ind = self.dataDims['tracking features dict'].index('crystal packing coefficient')
                self.crystal_density_ind = self.dataDims['tracking features dict'].index('crystal calculated density')

            # get batch size
            if config.auto_batch_sizing:
                print('Finding optimal batch size')
                train_loader, test_loader, config.final_batch_size = self.get_batch_size(dataset_builder, config)
            else:
                print('Getting dataloaders for pre-determined batch size')
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.max_batch_size

            print("Training batch size set to {}".format(config.final_batch_size))
            # model, optimizer, schedulers
            print('Reinitializing model and optimizer')
            if 'gan'.lower() in config.mode:
                generator, discriminator, g_optimizer, \
                g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)
                n_params = params1 + params2
            else:
                model, optimizer, schedulers, n_params = self.init_model(config, self.dataDims)

            # cuda
            if config.device.lower() == 'cuda':
                print('Putting model on CUDA')
                torch.backends.cudnn.benchmark = True
                # model = torch.nn.DataParallel(model) # send to multiple GPUs - not always working with wandb
                if 'gan'.lower() in config.mode:
                    generator.cuda()
                    discriminator.cuda()
                else:
                    model.cuda()

            if 'gan'.lower() in config.mode:
                wandb.watch((generator,discriminator), log_graph=True, log_freq = 100)
            else:
                wandb.watch(model, log_graph=True)

            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_metrics(config=config)

            # training loop
            d_hit_max_lr, g_hit_max_lr, converged, epoch = False, False, False, 0
            # if config.anomaly_detection:
            #     torch.autograd.set_detect_anomaly = True
            with torch.autograd.set_detect_anomaly(True):
                while (epoch < config.max_epochs) and not converged:
                    # very cool
                    print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                    print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                    print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                    # very cool
                    print("Starting Epoch {}".format(epoch))  # index from 0, very cool

                    if 'gan'.lower() in config.mode:
                        '''
                        train
                        '''
                        d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                            self.gan_epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                           g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                           update_gradients=True, record_stats=True)  # train & compute test loss

                        d_err_te, d_te_record, g_err_te, g_te_record, test_epoch_stats_dict, time_test = \
                            self.gan_epoch(config, dataLoader=test_loader, generator=generator, discriminator=discriminator,
                                           update_gradients=False, record_stats=True)  # compute loss on test set

                        print('epoch={}; d_nll_tr={:.5f}; d_nll_te={:.5f}; g_nll_tr={:.5f}; g_nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
                            epoch, torch.mean(torch.stack(d_err_tr)), torch.mean(torch.stack(d_err_te)),
                            torch.mean(torch.stack(g_err_tr)), torch.mean(torch.stack(g_err_te)),
                            time_train, time_test))

                        '''
                        update LR
                        '''
                        # update learning rate
                        g_optimizer = set_lr(g_schedulers, g_optimizer, config.generator.lr_schedule,
                                             config.generator.learning_rate, config.generator.max_lr,g_err_tr, g_hit_max_lr)
                        g_learning_rate = g_optimizer.param_groups[0]['lr']
                        if g_learning_rate >= config.generator.max_lr: g_hit_max_lr = True

                        # update learning rate
                        d_optimizer = set_lr(d_schedulers, d_optimizer, config.discriminator.lr_schedule,
                                             config.discriminator.learning_rate, config.discriminator.max_lr,d_err_tr, d_hit_max_lr)
                        d_learning_rate = d_optimizer.param_groups[0]['lr']
                        if d_learning_rate >= config.discriminator.max_lr: d_hit_max_lr = True

                        '''
                        logging
                        '''
                        # logging
                        metrics_dict = self.update_gan_metrics(
                            epoch, metrics_dict, d_err_tr, d_err_te,
                            g_err_tr, g_err_te, d_learning_rate, d_learning_rate)

                        self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict, d_tr_record, d_te_record, g_tr_record, g_te_record)
                        if epoch % config.wandb.sample_reporting_frequency == 0:
                            self.log_gan_accuracy(epoch, dataset_builder, train_loader, test_loader,
                                                  metrics_dict, g_tr_record, g_te_record, d_tr_record, d_te_record,
                                                  train_epoch_stats_dict, test_epoch_stats_dict, config,
                                                  generator, discriminator, wandb_log_figures=config.wandb.log_figures)

                        '''
                        convergence checks
                        '''

                        # check for convergence
                        if checkConvergence(metrics_dict['generator test loss'], config.history, config.generator.convergence_eps) and (epoch > config.history + 2):
                            config.finished = True
                            # self.log_gan_accuracy(epoch, dataset_builder, train_loader, test_loader,
                            #                   te_record, epoch_stats_dict,
                            #                   config, model, wandb_log_figures=True)  # always log figures at end of run
                            break

                    else:  # todo officially deprecate the old way of doing this #flow model will likely no longer function on its own here
                        err_tr, tr_record, time_train = \
                            self.model_epoch(config, dataLoader=train_loader, model=model,
                                             optimizer=optimizer, update_gradients=True)  # train & compute test loss

                        err_te, te_record, epoch_stats_dict, time_test = \
                            self.model_epoch(config, dataLoader=test_loader, model=model,
                                             update_gradients=False, record_stats=True)  # compute loss on test set

                        print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_train, time_test))

                        # update learning rate
                        optimizer = set_lr(schedulers, optimizer, config, err_tr, hit_max_lr)
                        learning_rate = optimizer.param_groups[0]['lr']
                        if learning_rate >= config.max_lr: hit_max_lr = True

                        # logging
                        self.update_metrics(epoch, metrics_dict, err_tr, err_te, learning_rate)
                        self.log_loss(metrics_dict, tr_record, te_record)
                        if epoch % config.wandb.sample_reporting_frequency == 0:
                            self.log_accuracy(epoch, dataset_builder, train_loader, test_loader,
                                              te_record, epoch_stats_dict,
                                              config, model, wandb_log_figures=config.wandb.log_figures)

                        # check for convergence
                        if checkConvergence(config, metrics_dict['test loss']) and (epoch > config.history + 2):
                            config.finished = True
                            self.log_accuracy(epoch, dataset_builder, train_loader, test_loader,
                                              te_record, epoch_stats_dict,
                                              config, model, wandb_log_figures=True)  # always log figures at end of run
                            break

                    epoch += 1

                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU



    def init_model(self, config, dataDims, print_status=True):
        '''
        Initialize model and optimizer
        :return:
        '''
        # init model
        print("Initializing model for " + config.mode)
        if config.mode == 'joint modelling':
            model = FlowModel(config, dataDims)
        elif 'molecule' in config.mode:
            model = molecule_graph_model(config, dataDims, crystal_mode=False)
        elif 'cell' in config.mode:
            model = molecule_graph_model(config, dataDims, crystal_mode=True)
        else:
            print(config.mode + ' is not a valid model mode!')
            sys.exit()

        if config.device == 'cuda':
            model = model.cuda()

        # init optimizers
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

        # init schedulers
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



    def log_loss(self, metrics_dict, tr_record, te_record):
        current_metrics = {}
        for key in metrics_dict.keys():
            current_metrics[key] = float(metrics_dict[key][-1])
            if 'loss' in key:  # log 'best' metrics
                current_metrics['best ' + key] = np.amin(metrics_dict[key])

            elif ('epoch' in key) or ('confusion' in key) or ('learning rate'):
                pass
            else:
                current_metrics['best ' + key] = np.amax(metrics_dict[key])

        for key in current_metrics.keys():
            current_metrics[key] = np.amax(current_metrics[key])  # just a formatting thing - nothing to do with the max of anything

        wandb.log(current_metrics)
        hist = np.histogram(tr_record, bins=256, range=(np.amin(tr_record), np.quantile(tr_record, 0.9)))
        wandb.log({"Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(te_record, bins=256, range=(np.amin(te_record), np.quantile(te_record, 0.9)))
        wandb.log({"Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        wandb.log({"Train Loss Coeff. of Variation": np.sqrt(np.var(tr_record)) / np.average(tr_record)})
        wandb.log({"Test Loss Coeff. of Variation": np.sqrt(np.var(te_record)) / np.average(te_record)})




    def get_dimension(self):

        if self.model_mode == 'joint modelling':
            dim = {
                'crystal features': self.lattice_keys,
                'n crystal features': len(self.lattice_keys),
                'dataset length': len(self.datapoints),
                'means': self.means,
                'stds': self.stds,
                'dtypes': self.dtypes,
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
            }
            if self.conditional_modelling:
                dim['conditional features'] = self.molecule_keys
                dim['n conditional features'] = len(self.molecule_keys)
                if self.conditioning_mode == 'graph model':  #
                    dim['output classes'] = [2]  # placeholder - will be overwritten later
                    dim['atom features'] = self.datapoints[0].x.shape[1]
                    dim['n mol features'] = self.n_mol_features
                    dim['atom embedding dict sizes'] = self.atom_dict_size
        elif 'regression' in self.model_mode:
            dim = {
                'atom features': self.datapoints[0].x.shape[1],
                'n mol features': self.n_mol_features,
                'output classes': [1],
                'dataset length': len(self.datapoints),
                'atom embedding dict sizes': self.atom_dict_size,
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
                'mean': self.mean,
                'std': self.std,
            }
        elif 'classification' in self.model_mode:
            dim = {
                'atom features': self.datapoints[0].x.shape[1],
                'n mol features': self.n_mol_features,
                'output classes': self.output_classes,
                'dataset length': len(self.datapoints),
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
                'class weights': self.class_weights,
                'class labels': self.class_labels,
            }
        elif self.model_mode == 'cell gan':
            dim = {
                'crystal features': self.lattice_keys,
                'n crystal features': len(self.lattice_keys),
                'dataset length': len(self.datapoints),
                'means': self.means,
                'stds': self.stds,
                'dtypes': self.dtypes,
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
                'atom features': self.datapoints[0].x.shape[1],
                'n mol features': self.n_mol_features,
                'mol features': self.mol_dict_keys,
                'atom embedding dict sizes': self.atom_dict_size,
                'conditional features': ['crystal system', 'crystal point group'],
                'n conditional features': 2
            }
            if self.conditional_modelling:
                dim['conditional features'] += self.molecule_keys
                dim['n conditional features'] += len(self.molecule_keys)

        else:
            print(self.model_mode + ' is not a valid mode!')
            sys.exit()

        dim['crystal system dict'] = self.crystal_system_dict
        dim['point group dict'] = self.point_group_dict
        dim['point groups to search'] = self.include_pgs

        return dim

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def __len__(self):
        return len(self.datapoints)

    '''
            # supercell method comparison
            supercell_data1, raw_generation, z_values = self.differentiable_generated_supercells(data, config, generator=generator, override_position = torch.ones((50,3))/4, override_orientation = torch.ones((50,3)) * torch.pi / 2)
            supercell_data2, raw_generation, z_values, generated_cell_volumes = self.fast_differentiable_generated_supercells(data, config, generator=generator,  override_position = torch.ones((50,3))/4, override_orientation = torch.ones((50,3)) * torch.pi / 2)
            mol1 = Atoms(numbers=supercell_data1.x[supercell_data1.batch==0,0].cpu().detach().numpy(),positions=supercell_data1.pos[supercell_data1.batch==0,:].cpu().detach().numpy())
            mol2 = Atoms(numbers=supercell_data2.x[supercell_data2.batch==0,0].cpu().detach().numpy(),positions=supercell_data2.pos[supercell_data2.batch==0,:].cpu().detach().numpy())
            view((mol1,mol2))


            real_data = data.y[0] * torch.tile(torch.tensor(config.dataDims['stds']).to(data.y[0].device),(len(data.y[0]),1)) + torch.tile(torch.tensor(config.dataDims['means']).to(data.y[0].device),(len(data.y[0]),1))
            real_lengths, real_angles, real_positions, real_orientations = torch.split(real_data.float(),3,dim=1)

            real_supercell_data = self.real_supercells(data.clone(), config)
            generated_samples = generator.forward(n_samples=data.num_graphs, conditions=data.clone().to(generator.device))
            fake_supercell_data, z_values, generated_cell_volumes = self.differentiable_generated_supercells(generated_samples, data.clone().to(generated_samples.device), config, 
                                                                                                             override_position = real_positions,
                                                                                                             override_orientation = real_orientations,
                                                                                                             override_cell_length = real_lengths,
                                                                                                             override_cell_angle = real_angles)
            ind = 0
            mol1 = Atoms(numbers=real_supercell_data.x[real_supercell_data.batch == ind, 0].cpu().detach().numpy(), positions=real_supercell_data.pos[real_supercell_data.batch == ind, :].cpu().detach().numpy())
            mol2 = Atoms(numbers=fake_supercell_data.x[fake_supercell_data.batch == ind, 0].cpu().detach().numpy(), positions=fake_supercell_data.pos[fake_supercell_data.batch == ind, :].cpu().detach().numpy())
            view((mol1, mol2))
            '''



    def real_supercells(self, supercell_data, config):
        '''
        test code for on-the-fly cell generation
        data = self.build_supercells(data)
        0. extract molecule and cell parameters
        1. find centroid
        2. find principal axis & angular component
        3. place centroid & align axes
        4. apply point symmetry
        5. tile supercell
        '''
        for i in range(supercell_data.num_graphs):  # todo speed this up
            # 0 extract molecule and cell parameters
            atoms = np.asarray(supercell_data.x[supercell_data.batch == i])
            # pre-enforce hydrogen cleanliness - shouldn't be necessary but you never know
            atomic_numbers = np.asarray(atoms[:, 0])  # this is currently atomic number - convert to masses
            heavy_inds = np.where(atomic_numbers != 1)
            atoms = atoms[heavy_inds]

            # get reference cell positions
            # first 3 columns are cartesian coords, last 3 are fractional

            cell_lengths = supercell_data.y[2][i][self.cell_length_inds]  # pull cell params from tracking inds
            cell_angles = supercell_data.y[2][i][self.cell_angle_inds]
            z_value = int(supercell_data.y[2][i][self.z_value_ind])
            T_fc = coor_trans_matrix('f_to_c', cell_lengths, cell_angles)
            cell_vectors = T_fc.dot(np.eye(3)).T

            reference_cell = supercell_data.y[3][i][:, :, :3]  # we're now pre-storing the packing rather than grabbing it from the CSD at runtime
            # #self.get_CSD_crystal(reader, csd_identifier=csd_identifier, mol_n_atoms=len(coords), z_value=z_value) # use CSD directly - slow
            # alternately, we could use the above random_coords functions with the known parameters to create the CSD cells (>99% accurate from a couple tests)

            supercell_atoms, supercell_coords = ref_to_supercell(reference_cell, z_value, atoms, cell_vectors)

            supercell_batch = torch.ones(len(supercell_atoms)).int() * i

            # append supercell info to the supercell_data class
            if i == 0:
                new_x = supercell_atoms
                new_coords = supercell_coords
                new_batch = supercell_batch
                new_ptr = torch.zeros(supercell_data.num_graphs)
            else:
                new_x = torch.cat((new_x, supercell_atoms), dim=0)
                new_coords = torch.cat((new_coords, supercell_coords), dim=0)
                new_batch = torch.cat((new_batch, supercell_batch))
                new_ptr[i] = new_ptr[-1] + len(new_x)

        # update dataloader with cell info
        supercell_data.x = new_x.type(dtype=torch.float32)
        supercell_data.pos = new_coords.type(dtype=torch.float32)
        supercell_data.batch = new_batch.type(dtype=torch.int64)
        supercell_data.ptr = new_ptr.type(dtype=torch.int64)

        return supercell_data




    def differentiable_generated_supercells(self, cell_sample, supercell_data, config, override_position=None, override_orientation=None, override_cell_length=None, override_cell_angle=None):
        '''
        convert cell parameters to reference cell
        convert reference cell to 3x3 supercell
        all using differentiable torch functions
        '''
        volumes = []
        z_values = []
        sg_numbers = [int(supercell_data.y[2][i][self.sg_number_ind]) for i in range(supercell_data.num_graphs)]
        lattices = [self.lattice_type[number] for number in sg_numbers]

        cell_lengths, cell_angles, rand_position, rand_rotation = cell_sample.split(3, 1)

        cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
            cell_lengths, cell_angles, rand_position, rand_rotation, lattices, config.dataDims, enforce_crystal_system=False)

        if override_position is not None:
            rand_position = torch.tensor(override_position).to(rand_position.device)
        if override_orientation is not None:
            rand_rotation = torch.tensor(override_orientation).to(rand_rotation.device)
        if override_cell_length is not None:
            cell_lengths = torch.tensor(override_cell_length).to(rand_position.device)
        if override_cell_angle is not None:
            cell_angles = torch.tensor(override_cell_angle).to(rand_rotation.device)

        for i in range(supercell_data.num_graphs):
            atoms = supercell_data.x[supercell_data.batch == i]
            atomic_numbers = atoms[:, 0]
            # heavy_atom_inds = torch.argwhere(atomic_numbers > 1)[:, 0]
            # assert torch.sum(atomic_numbers == 1) == 0, 'hydrogens in supercell_dataset!'
            # atoms = atoms_i[heavy_atom_inds]
            coords = supercell_data.pos[supercell_data.batch == i, :]
            weights = torch.tensor([self.atom_weights[int(number)] for number in atomic_numbers]).to(coords.device)

            sym_ops = torch.tensor(self.sym_ops[sg_numbers[i]], dtype=coords.dtype).to(coords.device)
            z_value = len(sym_ops)  # number of molecules in the reference cell
            z_values.append(z_value)

            T_fc, vol = coor_trans_matrix_torch('f_to_c', cell_lengths[i], cell_angles[i], return_vol=True)
            T_fc = T_fc.to(coords.device)
            T_cf = torch.linalg.inv(T_fc)  # faster #coor_trans_matrix_torch('c_to_f', cell_lengths[i], cell_angles[i]).to(config.device)
            cell_vectors = torch.inner(T_fc, torch.eye(3).to(coords.device)).T  # T_fc.dot(torch.eye(3)).T
            volumes.append(vol)

            random_coords = randomize_molecule_position_and_orientation_torch(
                coords, weights, T_fc, sym_ops,
                set_position=rand_position[i], set_rotation=rand_rotation[i])

            reference_cell = build_random_crystal_torch(T_cf, T_fc, random_coords, sym_ops, z_value)

            supercell_atoms, supercell_coords = ref_to_supercell_torch(reference_cell, z_value, atoms, cell_vectors)

            supercell_batch = torch.ones(len(supercell_atoms)).int() * i

            # append supercell info to the data class #
            if i == 0:
                new_x = supercell_atoms
                new_coords = supercell_coords
                new_batch = supercell_batch
                new_ptr = torch.zeros(supercell_data.num_graphs)
            else:
                new_x = torch.cat((new_x, supercell_atoms), dim=0)
                new_coords = torch.cat((new_coords, supercell_coords), dim=0)
                new_batch = torch.cat((new_batch, supercell_batch))
                new_ptr[i] = new_ptr[-1] + len(new_x)

        # update dataloader with cell info
        supercell_data.x = new_x.type(dtype=torch.float32)
        supercell_data.pos = new_coords.type(dtype=torch.float32)
        supercell_data.batch = new_batch.type(dtype=torch.int64)
        supercell_data.ptr = new_ptr.type(dtype=torch.int64)

        return supercell_data, z_values, volumes


def fast_differentiable_apply_point_symmetry(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values):
    '''
    apply point symmetries to single molecules
    NOTE: THIS IS BROKEN - IMPROPER POINT SYMMETRIES
    '''

    z_values = torch.tensor(z_values)
    unique_z_values = torch.unique(z_values)
    z_inds = [torch.where(z_values == z)[0] for z in unique_z_values]

    reference_cell_list_i = []

    for i, (inds, z_value) in enumerate(zip(z_inds, unique_z_values)):
        lens = torch.tensor([len(final_coords_list[ii]) for ii in inds])
        # pad = torch.max(lens) - lens
        # padded_coords_c = rnn.pad_sequence(final_coords_list, batch_first=True)[inds]
        padded_coords_f = torch.einsum('nij,nmj->nmi',
                                       (T_cf_list[inds],
                                        rnn.pad_sequence(final_coords_list, batch_first=True)[inds]))

        ref_cell = torch.zeros((z_value, len(inds), padded_coords_f.shape[1], 3)).to(final_coords_list[0].device)

        z_sym_ops = torch.stack([sym_ops_list[j] for j in inds])
        affine_points = torch.cat((padded_coords_f, torch.ones(padded_coords_f.shape[:-1] + (1,)).to(padded_coords_f.device)), dim=-1)

        for zv in range(z_value):
            dup_f = torch.einsum('nmj,nij->nmi', (affine_points, z_sym_ops[:, zv]))[:, :, :-1]
            # centroids_f_z = torch.einsum('nj,nij->ni',(affine_centroids_f, z_sym_ops[:,zv]))[:,:-1]
            centroids_f_z = torch.stack([dup_f[ii, :lens[ii]].mean(0) for ii in range(len(dup_f))])  # todo slow part
            # image_f = dup_f - torch.floor(centroids_f_z)[:,None,:]
            ref_cell[zv, :, :] = torch.einsum('nij,nmj->nmi', (T_fc_list[inds],
                                                               dup_f - torch.floor(centroids_f_z)[:, None, :]))

        # cells = [ref_cell[:,jj,:lens[jj],:] for jj in range(len(lens))]
        reference_cell_list_i.extend([ref_cell[:, jj, :lens[jj], :] for jj in range(len(lens))])

    sorted_z_inds = torch.argsort(torch.cat(z_inds))

    return [reference_cell_list_i[ind] for ind in sorted_z_inds]



def orig_fast_differentiable_apply_point_symmetry(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values):
    '''
    apply point symmetries to single molecules
    '''

    reference_cell_list = []
    for i, (coords, sym_ops, T_cf, T_fc, z_value) in enumerate(zip(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values)):
        coords_f = torch.inner(T_cf, coords).T
        affine_points = torch.cat((coords_f, torch.ones(coords_f.shape[:-1] + (1,)).to(coords.device)), dim=-1)

        ref_cell = torch.zeros((z_value, len(coords_f), 3)).to(coords.device)
        for zv in range(z_value):
            dup_f = torch.inner(affine_points, sym_ops[zv])[..., :-1]
            centroid_f = dup_f.mean(0)
            if (any(centroid_f < 0)) or (any(centroid_f > 1)):
                image_f = dup_f - torch.floor(centroid_f)
            else:
                image_f = dup_f
            ref_cell[zv] = torch.inner(T_fc, image_f).T

        reference_cell_list.append(ref_cell)

    reference_cell_list = []
    for i, (coords, sym_ops, T_cf, T_fc, z_value) in enumerate(zip(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values)):
        coords_f = torch.inner(T_cf, coords).T

        affine_points = torch.cat((coords_f, torch.ones(coords_f.shape[:-1] + (1,)).to(coords.device)), dim=-1)

        ref_cell = torch.zeros((z_value, len(coords_f), 3)).to(coords.device)
        for zv in range(z_value):
            dup_f = torch.inner(affine_points, sym_ops[zv])[..., :-1]
            ref_cell[zv] = torch.inner(T_fc, dup_f).T

        reference_cell_list.append(ref_cell)

    # accuracy check
    mats = torch.stack([torch.cdist(reference_cell_list[0][ii], reference_cell_list[0][ii], p=2) for ii in range(z_value)])

    diffs = [(mats[0] - mats[i]).abs().sum().cpu().detach().numpy() for i in range(len(mats))]

    print(np.sum(diffs))

    return reference_cell_list


def prev_fast_differentiable_apply_point_symmetry(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values):
    '''
    apply point symmetries to single molecules
    '''

    reference_cell_list = []
    for i, (coords, sym_ops, T_cf, T_fc, z_value) in enumerate(zip(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values)):
        coords_f = torch.inner(T_cf, coords).T
        affine_points = torch.cat((coords_f, torch.ones(coords_f.shape[:-1] + (1,)).to(coords.device)), dim=-1)

        ref_cell = torch.zeros((z_value, len(coords_f), 3)).to(coords.device)
        for zv in range(z_value):
            dup_f = torch.inner(affine_points, sym_ops[zv])[..., :-1]
            centroid_f = dup_f.mean(0)
            if (any(centroid_f < 0)) or (any(centroid_f > 1)):
                image_f = dup_f - torch.floor(centroid_f)
            else:
                image_f = dup_f
            ref_cell[zv] = torch.inner(T_fc, image_f).T

        reference_cell_list.append(ref_cell)

    return reference_cell_list
