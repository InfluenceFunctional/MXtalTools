'''import statements'''
import tqdm
import numpy as np
import torch

import constants.asymmetric_units
from models.utils import softmax_and_score
from models.vdw_overlap import vdw_overlap
from crystal_building.utils import \
    (random_crystaldata_alignment, align_crystaldata_to_principal_axes,
     batch_asymmetric_unit_pose_analysis_torch, DEPRECATED_write_sg_to_all_crystals)
from common.geometry_calculations import batch_molecule_principal_axes_torch, compute_Ip_handedness

'''
This script uses Markov Chain Monte Carlo, including the STUN algorithm, to optimize a given function
to-do- build option for steps to be proposed by gradient descent from the scoring model

> Inputs: model to be optimized over
> Outputs: 12D cell parameters

'''


class mcmcSampler:
    """
    finds optimum values of the function defined by the model
    intrinsically parallel, rather than via multiprocessing
    """

    def __init__(self,
                 gammas,
                 seedInd,
                 generator,
                 STUN_mode=False,
                 debug=False,
                 global_temperature=1,
                 init_adaptive_step_size=1,
                 supercell_size=5,
                 graph_convolution_cutoff=6,
                 vdw_radii=None,
                 preset_minimum=None,
                 reset_patience=1e3,
                 new_minimum_patience=1e3,
                 target_acceptance_rate=0.234,  # found this in a paper - optimal MCMC acceptance ratio
                 spacegroup_to_search='P-1',
                 conformer_orientation='random',
                 ):
        self.conformer_orientation = conformer_orientation
        self.STUN = STUN_mode  # modify the acceptance function with stochastic tunneling
        self.sg_to_search = spacegroup_to_search
        self.debug = debug
        self.target_acceptance_rate = target_acceptance_rate
        self.deltaIter = int(1)  # get outputs every this many of iterations with one iteration meaning one move proposed for each "particle" on average
        self.randintsResampleAt = int(1e4)  # how often to resample random numbers
        self.gammas = gammas
        self.nruns = len(gammas)
        self.global_temperature = global_temperature
        self.init_adaptive_step_size = init_adaptive_step_size
        self.adaptive_step_size = [self.init_adaptive_step_size for _ in range(self.nruns)]
        self.dim = 12
        self.generator = generator
        self.supercell_size = supercell_size
        self.graph_convolution_cutoff = graph_convolution_cutoff
        self.vdw_radii = vdw_radii
        self.preset_minimum = preset_minimum
        self.reset_patience = reset_patience
        self.new_minimum_patience = new_minimum_patience
        self.acceptance_history = self.new_minimum_patience  # trailing history over which to compute acceptance rate statistics
        self.move_size = 1  # manually rescale proposed move sizes

        np.random.seed(int(seedInd))  # initial seed is randomized over pipeline iterations
        torch.manual_seed(seedInd)

        if self.debug:
            self.initialize_debug_statistics()

    def __call__(self, model, builder, crystaldata, init_samples, iters, optimizer=None):
        self.resampled_state_record = [[0] for _ in range(self.nruns)]
        if init_samples is None:
            self.fresh_config_ind = 0
            self.resample_state(crystaldata, 0, reset_all=True)
        else:
            self.current_state = init_samples
        self.supercell_builder = builder
        self.run_sampling(model, crystaldata, iters)

        outputs = {
            "samples": np.stack(self.all_samples).T,
            "canonical samples": np.stack(self.all_samples_canonical).T,
            "scores": -np.stack(self.all_scores).T,  # reset correct score sign
            "vdw penalties": np.stack(self.all_vdw_penalties).T,
            "resampled state record": self.resampled_state_record,
        }
        if self.debug:
            outputs['step size'] = np.stack(self.adaptive_step_size_record)
            outputs['acceptance ratio'] = np.stack(self.acceptance_rate_record)
            outputs['stun score'] = np.stack(self.stun_score_record)

        return outputs

    def makeNewConfigs(self, crystaldata):
        '''
        :return:
        '''
        # crystaldata = self.align_crystaldata(crystaldata)
        if self.generator._get_name() == 'crystal_generator':
            return self.generator.forward(,.cpu().detach().numpy()
        else:
            return self.generator.forward(crystaldata.num_graphs, crystaldata,,.cpu().detach().numpy()

    def resample_state(self, crystaldata, ind, reset_all=False):
        """
        re-randomize a particular configuration
        :return:
        """
        if (self.fresh_config_ind == crystaldata.num_graphs) or (self.fresh_config_ind == 0):  # if we have run out of configs, make a fresh batch
            self.fresh_configs = self.makeNewConfigs(crystaldata)
            self.fresh_config_ind = 0

        if reset_all:
            self.current_state = np.copy(self.fresh_configs)
        else:
            self.current_state[ind, :] = np.copy(self.fresh_configs[self.fresh_config_ind, :])
            self.fresh_config_ind += 1
            self.resampled_state_record[ind].append(self.iter)

    def resample_random_numbers(self):
        """
        periodically resample our relevant random numbers
        :return:
        """
        # self.move_randns = np.random.normal(size=(self.nruns, self.randintsResampleAt)) * self.move_size  # randn move, equal for each dim (standardized)
        self.move_randns = np.random.normal(size=(self.nruns, self.randintsResampleAt)) * self.move_size  # * np.random.lognormal(size=(self.nruns, self.randintsResampleAt)) * self.move_size
        # self.move_randns = np.random.normal(size=(self.nruns, self.dim, self.randintsResampleAt)) * self.move_size # propose collective move

        # lognormal scaling increases both small and large steps, at cost of medium steps, equal for each dim (standardized)
        # todo pick only moves which are valid for a given space group

        self.pickDimRandints = np.random.choice([0, 1, 2, 4, 6, 8, 9, 10, 11], size=(self.nruns, self.randintsResampleAt))  # don't change alpha and gamma, for now
        # self.pickDimRandints = np.random.randint(6, self.dim, size=(self.nruns, self.randintsResampleAt))
        self.alphaRandoms = np.random.random((self.nruns, self.randintsResampleAt)).astype(float)

    def initialize_optima(self, scores, vdw_penalties):
        """
        initialize the minimum energies
        :return:
        """
        # trajectory
        self.all_scores = np.zeros((self.run_iters, self.nruns))  # record optima of the score function
        self.all_samples = np.zeros((self.run_iters, self.nruns, self.dim))
        self.all_samples_canonical = np.zeros((self.run_iters, self.nruns, self.dim))
        self.all_vdw_penalties = np.zeros_like(self.all_scores)
        self.accepted_inds = [[0] for _ in range(self.nruns)]
        self.new_optima_inds = [[] for i in range(self.nruns)]

        self.run_wise_best_scores = np.copy(scores)  # initialize the 'best score' value
        if self.preset_minimum is not None:
            self.minimum_found_score = -self.preset_minimum
        else:
            self.minimum_found_score = np.amin(self.run_wise_best_scores)

        self.all_scores[0] = scores
        self.all_samples[0] = self.current_state
        self.current_state_canonical = np.copy(self.proposed_state_canonical)
        self.all_samples_canonical[0] = np.copy(self.proposed_state_canonical)
        self.all_vdw_penalties[0] = vdw_penalties
        for i in range(self.nruns):
            self.new_optima_inds[i].append(0)

    def initialize_debug_statistics(self):
        '''
        step-by-step records for debugging purposes
        :return:
        '''
        self.adaptive_step_size_record = [[] for i in range(self.nruns)]
        self.acceptance_rate_record = [[] for i in range(self.nruns)]
        self.stun_score_record = [[] for i in range(self.nruns)]
        self.score_record = [[] for i in range(self.nruns)]
        self.vdw_overlap_record = [[] for i in range(self.nruns)]

    def initialize_convergence_statistics(self):
        # convergence stats
        self.resetInd = [0 for i in range(self.nruns)]  # flag
        self.running_acceptance_rate = np.zeros(self.nruns)  # rolling MCMC acceptance rate
        self.fresh_config_ind = 0

    def run_sampling(self, score_model, crystaldata, iters):
        """
        run the sampler until we converge to an optimum
        :return:
        """
        self.initialize_convergence_statistics()
        self.resample_random_numbers()
        self.run_iters = iters

        '''
        set symmetry info
        '''
        override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index(self.sg_to_search) + 1
        sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(crystaldata.x.device) for i in range(crystaldata.num_graphs)]
        crystaldata = DEPRECATED_write_sg_to_all_crystals(self.sg_to_search, self.supercell_builder.dataDims, crystaldata, self.supercell_builder.symmetries_dict, sym_ops_list)

        score_model = score_model.cuda()
        crystaldata = crystaldata.cuda()

        score_model.eval()
        for self.iter in tqdm.tqdm(range(self.run_iters), miniters=int(self.run_iters / 25)):  # sample for a certain number of iterations
            self.random_number_index = self.iter % self.randintsResampleAt  # random number index
            with torch.no_grad():  # random MCMC move
                self.iterate(score_model, crystaldata)  # try a monte-carlo step!

            if (self.iter % self.deltaIter == 0) and (self.iter > 0):  # every N iterations do some reporting / updating
                self.update_annealing_parameters(crystaldata)  # change temperature or other conditions

            if self.iter % self.randintsResampleAt == 0:  # periodically resample random numbers
                self.resample_random_numbers()

    def prop_random_configs(self):
        """
        propose a new ensemble of configurations
        :param ind:
        :return:
        """

        self.proposed_states = np.copy(self.current_state) if isinstance(self.current_state, np.ndarray) else torch.clone(self.current_state)
        # todo vectorize
        for i in range(self.nruns):  # rather than set an acceptance temperature, use it to modulate the step size
            self.proposed_states[i, self.pickDimRandints[i, self.random_number_index]] += self.move_randns[i, self.random_number_index] * self.adaptive_step_size[i]
            # self.proposed_states[i, :] = self.move_randns[i, :, self.random_number_index] * self.adaptive_step_size[i] # collective move

    def iterate(self, score_model, crystaldata):
        """
        run chainLength cycles of the sampler
        process: 1) propose state, 2) compute acceptance ratio, 3) sample against this ratio and accept/reject move
        :return: config, energy, and stun function will update
        """

        # propose a new state
        self.prop_random_configs()

        # even if it didn't change, just run it anyway (big parallel - to hard to disentangle)
        # compute acceptance ratio
        proposed_sample_scores, proposed_vdw_penalties, proposed_supercells = \
            self.score_proposed_samples(score_model, crystaldata)

        if self.iter == 0:  # initialize optima recording
            self.initialize_optima(proposed_sample_scores, proposed_vdw_penalties)
            self.current_scores = proposed_sample_scores
            self.current_vdw_penalties = proposed_vdw_penalties

        STUN_scores, score_differences = self.compute_score_difference([self.current_scores, proposed_sample_scores])
        acceptance_ratio = self.compute_acceptance_ratio(score_differences)

        self.update_current_state(acceptance_ratio, proposed_sample_scores, proposed_vdw_penalties)
        self.recordTrajectory()  # if we accept the move, update the trajectory
        if self.debug:  # record a bunch of detailed outputs
            self.record_debug_statistics()

    def align_crystaldata(self, crystaldata):
        if self.conformer_orientation == 'standardized':
            crystaldata = align_crystaldata_to_principal_axes(crystaldata)
        elif self.conformer_orientation == 'random':
            crystaldata = random_crystaldata_alignment(crystaldata)
            right_handed = True
            if right_handed:
                coords_list = [crystaldata.pos[crystaldata.ptr[i]:crystaldata.ptr[i + 1]] for i in range(crystaldata.num_graphs)]
                coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(crystaldata.num_graphs)]
                principal_axes_list, _, _ = batch_molecule_principal_axes_torch(coords_list_centred)
                handedness = compute_Ip_handedness(principal_axes_list)
                for ind, hand in enumerate(handedness):
                    if hand == -1:
                        crystaldata.pos[crystaldata.batch == ind] = -crystaldata.pos[crystaldata.batch == ind]  # invert

        return crystaldata

    def compute_acceptance_ratio(self, score_differences):
        return np.minimum(1, np.exp(-score_differences / self.global_temperature)) if isinstance(score_differences / self.global_temperature, np.ndarray) \
            else torch.min(torch.ones_like(score_differences), torch.exp(-score_differences))

    def update_current_state(self, acceptance_ratio, proposed_sample_scores, proposed_vdw_penalties):
        '''
        check Metropolis conditions, update configurations, and record statistics
        :return:
        '''
        # accept or reject
        # todo vectorize
        for i in range(self.nruns):
            if self.alphaRandoms[i, self.random_number_index] < acceptance_ratio[i]:  # accept
                self.current_state[i] = np.copy(self.proposed_states[i]) if isinstance(self.proposed_states[i], np.ndarray) else torch.clone(self.proposed_states[i])
                self.current_state_canonical[i] = np.copy(self.proposed_state_canonical[i]) if isinstance(self.proposed_state_canonical[i], np.ndarray) else torch.clone(self.proposed_state_canonical[i])
                self.current_scores[i] = proposed_sample_scores[i]
                self.current_vdw_penalties[i] = proposed_vdw_penalties[i]
                self.accepted_inds[i].append(self.iter)
                if proposed_sample_scores[i] < self.run_wise_best_scores[i]:
                    self.updateBest(i, proposed_sample_scores)

    def recordTrajectory(self):
        self.all_scores[self.iter] = self.current_scores
        self.all_samples[self.iter] = self.current_state
        self.all_samples_canonical[self.iter] = self.current_state_canonical
        self.all_vdw_penalties[self.iter] = self.current_vdw_penalties

    def compute_score_difference(self, scores):
        if self.STUN:  # compute score difference using STUN
            F = self.compute_STUN_score(scores)
            DE = F[1] - F[0]
        else:  # compute raw score difference
            F = [0, 0]
            DE = scores[1] - scores[0]

        return F, DE

    def compute_STUN_score(self, scores):
        """
        compute the STUN function for the given energies
        :return:
        """
        if isinstance(scores, np.ndarray):
            return 1 - np.exp(-self.gammas * (scores - self.minimum_found_score))  # compute STUN function with shared global minimum

        elif torch.is_tensor(scores):
            return 1 - torch.exp(-self.gammas * (scores - self.minimum_found_score))  # compute STUN function with shared global minimum

        else:
            assert False, 'score must be a numpy array or torch tensor'

    def record_debug_statistics(self):
        for i in range(self.nruns):  # todo vectorize
            self.adaptive_step_size_record[i].append(self.adaptive_step_size[i])
            self.acceptance_rate_record[i].append(self.running_acceptance_rate[i])
            self.score_record[i].append(self.current_scores[i])
            if self.STUN:
                self.stun_score_record[i].append(self.STUN_scores[i])

            self.vdw_overlap_record[i].append(self.current_vdw_penalties)

    def score_proposed_samples(self, score_model, crystaldata):
        """
        DEPRECATED
        compute score against which we're optimizing
        :param prop_config:
        :param config:
        :return:
        """
        proposed_supercells, _ = self.supercell_builder.build_zp1_supercells(crystaldata, torch.Tensor(self.proposed_states),
                                                                             supercell_size=self.supercell_size,
                                                                             graph_convolution_cutoff=self.graph_convolution_cutoff,
                                                                             override_sg=self.sg_to_search,
                                                                             align_to_standardized_orientation=False, )

        output, dist_dict = score_model(proposed_supercells.clone().cuda(), return_dists=True)

        vdw_penalty = vdw_overlap(self.vdw_radii, dists=dist_dict['dists_dict']['intermolecular_dist'],
                                  atomic_numbers=dist_dict['dists_dict']['intermolecular_dist_atoms'],
                                  batch_numbers=dist_dict['dists_dict']['intermolecular_dist_batch'],
                                  num_graphs=crystaldata.num_graphs).cpu().detach().numpy()

        score = -softmax_and_score(output).cpu().detach().numpy()  # we want actually to maximize the score
        # todo get the standardized canonical orienattion
        correct_position = np.zeros((proposed_supercells.num_graphs, 3))
        correct_rotation = np.zeros((proposed_supercells.num_graphs, 3))
        for jj in range(proposed_supercells.num_graphs):  # all assuming fully right handed
            correct_position[jj], correct_rotation[jj], handedness \
                = asymmetric_unit_pose_analysis_np(proposed_supercells.ref_cell_pos[jj],
                                                   proposed_supercells.sg_ind[jj],
                                                   constants.asymmetric_units.asym_unit_dict,
                                                   torch.linalg.inv(proposed_supercells.T_fc[jj]),
                                                   enforce_right_handedness = False) # todo replace this with the raw cell params
        # renormalize
        nonstandardized_state = proposed_supercells.cell_params.cpu().detach().numpy()
        nonstandardized_state[:, -3:] = correct_rotation
        self.proposed_state_canonical = (nonstandardized_state - self.supercell_builder.dataDims['lattice_means']) / self.supercell_builder.dataDims['lattice_stds']

        return score, vdw_penalty, proposed_supercells

    def updateBest(self, ind, proposed_sample_scores):
        self.run_wise_best_scores[ind] = proposed_sample_scores[ind]
        self.new_optima_inds[ind].append(self.iter)
        if self.run_wise_best_scores[ind] < self.minimum_found_score:
            self.minimum_found_score = self.run_wise_best_scores[ind]

    def update_annealing_parameters(self, crystaldata):
        """
        Following "Adaptation in stochatic tunneling global optimization of complex potential energy landscapes"
        1) updates temperature according to STUN threshold to separate "Local search" and "tunneling" phases
        2) determines when the algorithm is no longer efficiently searching and adapts by resetting the config to a random value
        """
        # 1) if rejection rate is too high, switch to tunneling mode, if it is too low, switch to local search mode
        # acceptanceRate = len(self.stunRec)/self.iter # global acceptance rate

        for i in range(self.nruns):
            acceptedRecently = np.sum((self.iter - np.asarray(self.accepted_inds[i][-self.acceptance_history:])) < self.acceptance_history)  # rolling acceptance rate - how many accepted out of the last hundred iters
            self.running_acceptance_rate[i] = acceptedRecently / self.acceptance_history

            if self.running_acceptance_rate[i] < self.target_acceptance_rate:  # for temperature as step size, we increase when acceptance rate is too high
                self.adaptive_step_size[i] = self.adaptive_step_size[i] * (1 - 0.1 * np.random.random(1)[0])  # modulate temperature semi-stochastically
            else:
                self.adaptive_step_size[i] = self.adaptive_step_size[i] * (1 + 0.1 * np.random.random(1)[0])

            # if self.adaptive_step_size[i] < 1e-5:  # if the step size gets too small, simply reset
            #     self.resetInd[i] = self.iter
            #     self.resample_state(crystaldata, i)  # re-randomize selected trajectories
            #     self.adaptive_step_size[i] = self.init_adaptive_step_size  # reset temperature
            #     self.accepted_inds[i] = []  # reset acceptance stats
            #     self.current_scores[i] = 100  # to avoid recomputing, just set it high so the next step will automatically be accpted

            # if we haven't found a new minimum in a long time, randomize input and do a temperature boost
            if (self.iter - self.resetInd[i]) > self.reset_patience:  # within xx of the last reset
                if (self.iter - self.accepted_inds[i][-1]) > self.new_minimum_patience:  # haven't seen made a new step in xx steps
                    self.resetInd[i] = self.iter
                    self.resample_state(crystaldata, i)  # re-randomize selected trajectories
                    self.adaptive_step_size[i] = self.init_adaptive_step_size  # reset temperature
                    self.accepted_inds[i] = []  # reset acceptance stats
                    self.current_scores[i] = 100  # to avoid recomputing, just set it high so the next step will automatically be accpted
