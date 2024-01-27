import os

import numpy as np
import torch
from tqdm import tqdm

from bulk_molecule_classification.classifier_constants import form2index, identifier2form
from bulk_molecule_classification.dump_data_processing import generate_dataset_from_dumps

from bulk_molecule_classification.traj_analysis_figs import embedding_fig, form_accuracy_fig, defect_accuracy_fig, all_accuracy_fig, classifier_trajectory_analysis_fig
from bulk_molecule_classification.utils import get_loss, classifier_reporting, record_step_results, process_trajectory_results_dict
from bulk_molecule_classification.dataset_prep import collect_to_traj_dataloaders


def train_classifier(config, classifier, optimizer,
                     train_loader, test_loader,
                     num_epochs, wandb,
                     class_names, ordered_class_names, device,
                     batch_size, reporting_frequency,
                     runs_path, run_name):
    with wandb.init(project='cluster_classifier', entity='mkilgour', config=config):
        wandb.run.name = run_name
        wandb.log({key: value for key, value in config.items()})
        test_record = []
        time_since_best = 0
        for epoch in range(num_epochs):
            print(f"starting epoch {epoch}")
            wandb.log({'epoch': epoch})

            train_loss = []
            train_true_labels = []
            train_probs = []
            train_true_defects = []
            test_loss = []
            test_probs = []
            test_true_labels = []
            test_true_defects = []
            classifier.train(True)
            optimizer.zero_grad()
            for step, sample in enumerate(tqdm(train_loader)):
                sample = sample.to(device)

                output = classifier(sample)
                loss = get_loss(output, sample, config['num_forms'])
                loss.backward()
                if step % batch_size == 0:  # use gradient accumulation for synthetically larger batch sizes
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss.append(loss.cpu().detach().numpy())
                train_probs.append(output.cpu().detach().numpy())
                train_true_labels.append(sample.y.cpu().detach().numpy())
                train_true_defects.append(sample.defect.cpu().detach().numpy())

            with torch.no_grad():
                classifier.eval()

                for step, sample in enumerate(tqdm(test_loader)):
                    sample = sample.to(device)

                    output = classifier(sample)  # fix mini-batch behavior
                    loss = get_loss(output, sample, config['num_forms'])

                    test_loss.append(loss.cpu().detach().numpy())
                    test_probs.append(output.cpu().detach().numpy())
                    test_true_labels.append(sample.y.cpu().detach().numpy())
                    test_true_defects.append(sample.defect.cpu().detach().numpy())

            test_record.append(np.mean(test_loss))
            if test_record[-1] == np.amin(test_record):
                torch.save({'model_state_dict': classifier.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           runs_path + run_name + '_best_classifier_checkpoint')

                time_since_best = 0
            else:
                time_since_best += 1

            if time_since_best > config['convergence_history']:
                break  # stop training if we are not improving test loss

            print(f"Log Train Loss {np.log10(np.mean(np.array(train_loss))):.4f}")
            print(f"Log Test Loss {np.log10(np.mean(np.array(test_loss))):.4f}")

            if epoch % reporting_frequency == 0:
                train_true_labels = np.concatenate(train_true_labels)
                train_true_defects = np.concatenate(train_true_defects)
                train_probs = np.concatenate(train_probs)
                test_true_labels = np.concatenate(test_true_labels)
                test_true_defects = np.concatenate(test_true_defects)
                test_probs = np.concatenate(test_probs)

                classifier_reporting(train_true_labels, train_true_defects, train_probs, class_names, ordered_class_names, wandb, 'Train')
                classifier_reporting(test_true_labels, test_true_defects, test_probs, class_names, ordered_class_names, wandb, 'Test')

            wandb.log({
                'train_loss': np.asarray(train_loss).mean(),
                'test_loss': np.asarray(test_loss).mean()
            })


def classifier_evaluation(config, classifier, train_loader, test_loader,
                          wandb, class_names, ordered_class_names, device, run_name):
    with wandb.init(project='cluster_classifier', entity='mkilgour'):
        wandb.run.name = run_name + '_evaluation'
        wandb.log({'config': config})

        results_dict = None
        classifier.train(False)
        with torch.no_grad():
            classifier.eval()
            for step, data in enumerate(tqdm(train_loader)):
                sample = data.to(device)
                (output, latents), embeddings = classifier(sample, return_latent=True, return_embedding=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents, embeddings, step, config)

            for step, data in enumerate(tqdm(test_loader)):
                sample = data.to(device)
                (output, latents), embeddings = classifier(sample, return_latent=True, return_embedding=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents, embeddings, step, config, index_offset=len(train_loader))

        for key in results_dict.keys():
            try:
                results_dict[key] = np.stack(results_dict[key])
            except:
                results_dict[key] = np.concatenate(results_dict[key], axis=0)

        results_dict['Atomwise_Sample_Index'] = results_dict['Sample_Index'].repeat(config['mol_num_atoms'])
        num_samples = len(results_dict['Targets'])

        fig_dict = {}
        '''embed train, test, melt'''
        fig_dict['Embedding_Analysis'], results_dict['embedding'] = embedding_fig(
            results_dict, num_samples, class_names, ordered_class_names, config['training_temps'],
            max_samples=1000, perplexity=30)

        '''accuracy metrics'''
        fig_dict['Form_Confusion_Matrices'], accuracy_scores = form_accuracy_fig(results_dict, ordered_class_names, config['training_temps'])
        fig_dict['Defect_Confusion_Matrices'], accuracy_scores = defect_accuracy_fig(results_dict, config['training_temps'])
        fig_dict['All_Confusion_Matrices'], accuracy_scores = all_accuracy_fig(results_dict, ordered_class_names, config['training_temps'])

        os.chdir(config['results_path'])
        [fig_dict[key].write_image(f"{config['run_name']}_Figure_{i}.png") for i, key in enumerate(fig_dict.keys())]
        results_dict['config'] = config
        np.save(config['run_name'] + '_evaluation_results_dict', results_dict)
        wandb.log(fig_dict)


def trajectory_analysis(config, classifier, wandb, device, dumps_dir):

    from bulk_molecule_classification.ovito_utils import write_ovito_xyz
    dataset_name = '_'.join(dumps_dir.split('/')[-3:])
    datasets_path = config['datasets_path']
    dataset_path = f'{datasets_path}{dataset_name}.pkl'
    output_dict_path = config['results_path'] + dataset_name + '_analysis'

    if True:  # not os.path.exists(output_dict_path + '.npy'):
        if not os.path.exists(dataset_path):
            made_dataset = generate_dataset_from_dumps([dumps_dir], dataset_path)

            if not made_dataset:
                print(f'{dumps_dir} does not contain valid dump to analyze')
                return False

        os.chdir(config['runs_path'])  # todo move this

        if os.path.exists(dumps_dir + 'run_config.npy'):
            run_config = np.load(dumps_dir + 'run_config.npy', allow_pickle=True).item()
        else:
            run_config = None

        _, loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                dataset_path, config['dataset_size'],
                                                batch_size=1, temperatures=None,
                                                conv_cutoff=config['conv_cutoff'],
                                                test_fraction=1, shuffle=False, filter_early=False,
                                                early_only=False,
                                                run_config=run_config,
                                                pare_to_cluster=False if 'interface' in dataset_path else True,
                                                single_trajectory=True)
        results_dict = None
        classifier.train(False)
        with torch.no_grad():
            classifier.eval()
            for step, data in enumerate(tqdm(loader)):
                sample = data.to(device)
                (output, latents), embeddings = classifier(sample, return_latent=True, return_embedding=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents, embeddings, step, config)

        for key in results_dict.keys():
            try:
                results_dict[key] = np.concatenate(results_dict[key], axis=0)
            except:
                results_dict[key] = np.stack(results_dict[key])

        os.chdir(config['results_path'])

        results_dict['Atomwise_Sample_Index'] = results_dict['Sample_Index'].repeat(config['mol_num_atoms'])
        results_dict['Type_Prediction_Choice'] = np.argmax(results_dict['Type_Prediction'], axis=-1)  # argmax sample
        results_dict['Type_Prediction_Choice'] = np.asarray([form2index[tgt] for tgt in results_dict['Type_Prediction_Choice']])
        results_dict['Type_Prediction_Confidence'] = np.amax(results_dict['Type_Prediction'], axis=-1)  # argmax sample

        sorted_molwise_results_dict, time_steps = process_trajectory_results_dict(results_dict, loader, config['mol_num_atoms'])
        del results_dict

        if dumps_dir == r'D:/crystals_extra/classifier_training/urea_melt_interface_T200':  # collect predictions to form I and IV or 'other'
            interface_mode = True
            for step in tqdm(range(len(sorted_molwise_results_dict['Molecule_Type_Prediction']))):
                orig_pred = sorted_molwise_results_dict['Molecule_Type_Prediction'][step]
                unknowns = orig_pred[:, [0, 1, 2, 4, 6]].sum(1)[:, None]
                new_pred = np.concatenate([orig_pred[:, [3, 5]], unknowns], axis=1)
                sorted_molwise_results_dict['Molecule_Type_Prediction'][step] = new_pred

                orig_choice = sorted_molwise_results_dict['Molecule_Type_Prediction_Choice'][step]
                orig_choice[orig_choice == 2] = 1  # form IV to index 1
                for ind in range(8):
                    if ind != 0 and ind != 1:
                        orig_choice[orig_choice == ind] = 2

                sorted_molwise_results_dict['Molecule_Type_Prediction_Choice'][step] = orig_choice
        else:
            interface_mode = False

        np.save(output_dict_path, sorted_molwise_results_dict)

        write_ovito_xyz(sorted_molwise_results_dict['Coordinates'],
                        sorted_molwise_results_dict['Atom_Types'],
                        sorted_molwise_results_dict['Molecule_Type_Prediction_Choice'], filename=dataset_name + '_prediction')  # write a trajectory

        write_ovito_xyz(sorted_molwise_results_dict['Coordinates'],
                        sorted_molwise_results_dict['Atom_Types'],
                        sorted_molwise_results_dict['Molecule_Type_Prediction'], filename=dataset_name + '_all_probs')  # write a trajectory

        if os.path.exists(dumps_dir + 'run_config.npy'):
            run_config = np.load(dumps_dir + 'run_config.npy', allow_pickle=True).item()
            if 'crystal' in dumps_dir and 'melt' in dumps_dir:
                inside_radius = run_config['max_sphere_radius']
            else:
                inside_radius = None
        else:
            inside_radius = None

        fig, fig2, traj_dict = classifier_trajectory_analysis_fig(
            sorted_molwise_results_dict, time_steps,
            'urea' if config['mol_num_atoms'] == 8 else 'nicotinamide',
            inside_radius=inside_radius, interface_mode=interface_mode)

        """ type density vs radius over time
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from scipy.ndimage import gaussian_filter1d

        timepoints = np.linspace(0, 200, 24).astype(int)
        
        fig = make_subplots(rows=4, cols=6, subplot_titles=[f't={timepoint}' for timepoint in timepoints])
        for ind in range(24):
            row = ind // 6 + 1
            col = ind % 6 + 1
            ind = timepoints[ind]
            print(row)
            print(col)
            radii = sorted_molwise_results_dict['Centroid Radii'][ind]
            f4_frac = sorted_molwise_results_dict['Molecule_Type_Prediction'][ind][:, 3]
            sort_inds = np.argsort(radii)[::-1]
            fig.add_trace(go.Scatter(x=radii[sort_inds], y=gaussian_filter1d(f4_frac[sort_inds],sigma=5)), row=row, col=col)
        fig.update_xaxes(title='Radius')
        fig.update_yaxes(title='Form 4 Fraction')
        fig.update_yaxes(range=[-0.05, 1.05])
        fig.show()
        """

        if os.path.exists(dumps_dir + 'run_config.npy'):
            fig.update_layout(
                title=f"Form {identifier2form[run_config['structure_identifier']]}, "
                      f"Cluster Radius {run_config['max_sphere_radius']}A, "
                      f"Temperature {run_config['temperature']}K")
            fig2.update_layout(
                title=f"Form {identifier2form[run_config['structure_identifier']]}, "
                      f"Cluster Radius {run_config['max_sphere_radius']}A, "
                      f"Temperature {run_config['temperature']}K")

        elif 'urea_interface' in dumps_dir:
            fig.update_layout(
                title="Urea Interface")
            run_config = None
        else:
            run_config = None
            print("Missing trajectory config")

        traj_analysis = traj_dict
        traj_analysis['run_config'] = run_config
        traj_analysis['eval_config'] = config

        fig.write_image(f"{dataset_name}_Trajectory_Analysis.png", scale=4)
        fig2.write_image(f"{dataset_name}_Combined_Trajectory_Analysis.png", scale=4)
        wandb.log({f"{dataset_name} Trajectory Analysis": fig})
