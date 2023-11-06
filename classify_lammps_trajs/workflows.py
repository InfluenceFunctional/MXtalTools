import os

import numpy as np
import torch
from tqdm import tqdm

from classify_lammps_trajs.NICOAM_constants import form2index, identifier2form
from classify_lammps_trajs.dump_data_processing import generate_dataset_from_dumps

from classify_lammps_trajs.traj_analysis_figs import embedding_fig, form_accuracy_fig, defect_accuracy_fig, all_accuracy_fig, classifier_trajectory_analysis_fig
from classify_lammps_trajs.utils import get_loss, classifier_reporting, record_step_results, collect_to_traj_dataloaders, process_trajectory_results_dict


def train_classifier(config, classifier, optimizer,
                     train_loader, test_loader,
                     num_epochs, wandb,
                     class_names, device,
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
            for step, data in enumerate(tqdm(train_loader)):
                sample = data.to(device)

                output = classifier(sample)
                loss = get_loss(output, sample)
                train_probs.append(output.cpu().detach().numpy())
                train_true_labels.append(sample.y.cpu().detach().numpy())
                train_true_defects.append(sample.defect.cpu().detach().numpy())

                loss.backward()
                if step % batch_size == 0:  # use gradient accumulation for synthetically larger batch sizes
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss.append(loss.cpu().detach().numpy())

            with torch.no_grad():
                classifier.eval()

                for step, data in enumerate(tqdm(test_loader)):
                    sample = data.to(device)

                    output = classifier(sample)  # fix mini-batch behavior
                    loss = get_loss(output, sample)

                    test_loss.append(loss.cpu().detach().numpy())
                    test_probs.append(output.cpu().detach().numpy())
                    test_true_labels.append(sample.y.cpu().detach().numpy())
                    test_true_defects.append(sample.defect.cpu().detach().numpy())

            test_record.append(np.mean(test_loss))
            if test_record[-1] == np.amin(test_record):
                torch.save({'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
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

                classifier_reporting(train_true_labels, train_true_defects, train_probs, class_names, wandb, 'Train')
                classifier_reporting(test_true_labels, test_true_defects, test_probs, class_names, wandb, 'Test')

            wandb.log({
                'train_loss': np.asarray(train_loss).mean(),
                'test_loss': np.asarray(test_loss).mean()
            })


def classifier_evaluation(config, classifier, optimizer,
                          train_loader, test_loader,
                          num_epochs, wandb,
                          class_names, device,
                          batch_size, reporting_frequency,
                          runs_path, run_name):
    with wandb.init(project='cluster_classifier', entity='mkilgour'):
        wandb.run.name = run_name + '_evaluation'
        wandb.log({'config': config})

        results_dict = None
        classifier.train(False)
        with torch.no_grad():
            classifier.eval()
            for step, data in enumerate(tqdm(train_loader)):
                sample = data.to(device)
                output, latents_dict = classifier(sample, return_latent=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents_dict, step)

            for step, data in enumerate(tqdm(test_loader)):
                sample = data.to(device)
                output, latents_dict = classifier(sample, return_latent=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents_dict, step, index_offset=len(train_loader))

        for key in results_dict.keys():
            try:
                results_dict[key] = np.stack(results_dict[key])
            except:
                results_dict[key] = np.concatenate(results_dict[key], axis=0)

        results_dict['Atomwise_Sample_Index'] = results_dict['Sample_Index'].repeat(15)
        num_samples = len(results_dict['Targets'])

        fig_dict = {}
        '''embed train, test, melt'''
        fig_dict['Embedding_Analysis'] = embedding_fig(results_dict, num_samples)

        '''accuracy metrics'''
        fig_dict['Form_Confusion_Matrices'], accuracy_scores = form_accuracy_fig(results_dict)
        fig_dict['Defect_Confusion_Matrices'], accuracy_scores = defect_accuracy_fig(results_dict)
        fig_dict['All_Confusion_Matrices'], accuracy_scores = all_accuracy_fig(results_dict)

        '''model games'''

        '''training curves'''

        os.chdir(config['runs_path'])
        # [fig_dict[key].write_image(f'Figure_{i}') for i, key in enumerate(fig_dict.keys())]
        wandb.log(fig_dict)


def trajectory_analysis(config, classifier, run_name, wandb, device, dumps_dir):


    from classify_lammps_trajs.ovito_utils import write_ovito_xyz
    dataset_name = '_'.join(dumps_dir.split('/')[-3:])
    datasets_path = config['datasets_path']
    dataset_path = f'{datasets_path}{dataset_name}.pkl'
    output_dict_path = config['runs_path'] + dataset_name + '_analysis'

    if not os.path.exists(output_dict_path + '.npy'):
        if not os.path.exists(dataset_path):
            made_dataset = generate_dataset_from_dumps([dumps_dir], dataset_path)

            if not made_dataset:
                print(f'{dumps_dir} does not contain valid dump to analyze')
                return False

        os.chdir(config['runs_path'])

        _, loader = collect_to_traj_dataloaders(
            dataset_path, int(250), batch_size=1, temperatures=None, test_fraction=1, shuffle=False, filter_early=False)

        results_dict = None
        classifier.train(False)
        with torch.no_grad():
            classifier.eval()
            for step, data in enumerate(tqdm(loader)):
                sample = data.to(device)
                output, latents_dict = classifier(sample, return_latent=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents_dict, step)

        for key in results_dict.keys():
            try:
                results_dict[key] = np.concatenate(results_dict[key], axis=0)
            except:
                results_dict[key] = np.stack(results_dict[key])

        results_dict['Atomwise_Sample_Index'] = results_dict['Sample_Index'].repeat(15)
        results_dict['Type_Prediction_Choice'] = np.argmax(results_dict['Type_Prediction'], axis=-1)  # argmax sample
        results_dict['Type_Prediction_Choice'] = np.asarray([form2index[tgt] for tgt in results_dict['Type_Prediction_Choice']])
        results_dict['Type_Prediction_Confidence'] = np.amax(results_dict['Type_Prediction'], axis=-1)  # argmax sample

        sorted_molwise_results_dict, time_steps = process_trajectory_results_dict(results_dict, loader)

        write_ovito_xyz(sorted_molwise_results_dict['Coordinates'],
                        sorted_molwise_results_dict['Atom_Types'],
                        sorted_molwise_results_dict['Type_Prediction_Choice'], filename=dataset_name + '_prediction')  # write a trajectory

        write_ovito_xyz(sorted_molwise_results_dict['Coordinates'],
                        sorted_molwise_results_dict['Atom_Types'],
                        sorted_molwise_results_dict['Type_Prediction_Confidence'], filename=dataset_name + '_confidence')  # write a trajectory

        fig, traj_dict = classifier_trajectory_analysis_fig(sorted_molwise_results_dict, time_steps)

        run_config = np.load(dumps_dir + 'run_config.npy', allow_pickle=True).item()
        fig.update_layout(
            title=f"Form {identifier2form[run_config['structure_identifier']]}, "
                  f"Cluster Radius {run_config['max_sphere_radius']}A, "
                  f"Temperature {run_config['temperature']}K")

        fig.update_yaxes(range=[0, 1])

        traj_analysis = traj_dict
        traj_analysis['run_config'] = run_config
        traj_analysis['eval_config'] = config

        np.save(output_dict_path, traj_analysis)

        wandb.log({f"{dataset_name} Trajectory Analysis": fig})