'''import statements'''
import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from utils import softmax_and_score
from models.vdw_overlap import vdw_overlap

'''
This script uses Markov Chain Monte Carlo, including the STUN algorithm, to optimize a given function

> Inputs: model to be optimized over
> Outputs: 12D cell parameters

'''


class Sampler:
    """
    finds optimum values of the function defined by the model
    intrinsically parallel, rather than via multiprocessing
    """

    def __init__(self,
                 gammas,
                 seedInd,
                 random_generator,
                 acceptance_mode='stun',
                 debug=False,
                 init_temp=1,
                 move_size=0.01,
                 supercell_size=1,
                 graph_convolution_cutoff=6,
                 vdw_radii=None,
                 preset_minimum=None,
                 reset_patience = 1e3,
                 new_minimum_patience = 1e3,
                 ):
        if acceptance_mode == 'stun':
            self.STUN = 1
        else:
            self.STUN = 0

        self.debug = debug
        self.target_acceptance_rate = 0.234  # found this in a paper - optimal MCMC acceptance ratio
        self.acceptance_history = 100
        self.deltaIter = int(1)  # get outputs every this many of iterations with one iteration meaning one move proposed for each "particle" on average
        self.randintsResampleAt = int(1e4)  # larger takes up more memory but increases speed
        self.gammas = gammas
        self.nruns = len(gammas)
        self.temp0 = init_temp  # initial temperature for sampling runs
        self.temperature = [self.temp0 for _ in range(self.nruns)]
        self.dim = 12
        self.generator = random_generator
        self.move_size = move_size
        self.supercell_size = supercell_size
        self.graph_convolution_cutoff = graph_convolution_cutoff
        self.vdw_radii = vdw_radii
        self.preset_minimum = preset_minimum
        self.reset_patience = reset_patience
        self.new_minimum_patience = new_minimum_patience

        np.random.seed(int(seedInd))  # initial seed is randomized over pipeline iterations

        if self.debug:
            self.initRecs()

    def __call__(self, model, builder, crystaldata, init_samples, iters):
        self.config = init_samples
        self.converge(model, builder, crystaldata, iters)

        outputs = {
            "samples": np.stack(self.all_samples).T,
            "scores": np.stack(self.all_scores).T,
            "energies": np.stack(self.all_energies).T,
            "uncertainties": np.stack(self.all_uncertainties).T,
            "vdw penalties": np.stack(self.all_vdw_penalties).T,
        }
        assert False # save samples for later reconstruction
        if self.debug:
            outputs['temperature'] = np.stack(self.temprec)
            outputs['acceptance ratio'] = np.stack(self.accrec)
            outputs['stun score'] = np.stack(self.stunrec)

            # outputs['raw score record']=np.stack(self.scorerec)
            # outputs['energy record']=np.stack(self.enrec)
            # outputs['std dev record']=np.stack(self.std_devrec)
        return outputs

    def makeAConfig(self, crystaldata):
        '''
        :return:
        '''
        assert 1 == 2 # set this up for our new generator
        return self.generator.forward(crystaldata, num_samples=crystaldata.num_graphs).cpu().detach().numpy()

    def resetConfig(self, crystaldata, ind):
        """
        re-randomize a particular configuration
        :return:
        """
        self.config[ind, :] = self.makeAConfig(crystaldata)[0, :]

    def resampleRandints(self):
        """
        periodically resample our relevant random numbers
        :return:
        """
        # self.move_randns = np.random.normal(size=(self.nruns, self.randintsResampleAt)) * self.move_size  # randn move, equal for each dim (standardized)
        self.move_randns = np.random.normal(size=(self.nruns, self.randintsResampleAt)) * np.random.lognormal(size=(self.nruns, self.randintsResampleAt)) * self.move_size  # lognormal scaling increases both small and large steps
        # , at cost of medium steps, equal for each dim (standardized)

        self.pickDimRandints = np.random.choice([0, 1, 2, 4, 6, 8, 9, 10, 11], size=(self.nruns, self.randintsResampleAt))  # don't change alpha and gamma, for now
        # self.pickDimRandints = np.random.randint(6, self.dim, size=(self.nruns, self.randintsResampleAt))
        self.alphaRandoms = np.random.random((self.nruns, self.randintsResampleAt)).astype(float)

    def initOptima(self, scores, energy, std_dev):
        """
        initialize the minimum energies
        :return:
        """
        # trajectory
        self.all_scores = np.zeros((self.run_iters, self.nruns))  # record optima of the score function
        self.all_energies = np.zeros_like(self.all_scores)  # record energies near the optima
        self.all_uncertainties = np.zeros_like(self.all_scores)  # record of uncertainty at the optima
        self.all_samples = np.zeros((self.run_iters, self.nruns, self.dim))
        self.all_vdw_penalties = np.zeros_like(self.all_scores)

        # optima
        self.new_optima_inds = [[] for i in range(self.nruns)]
        self.recInds = [[] for i in range(self.nruns)]
        self.new_optima_scores = [[] for i in range(self.nruns)]  # new minima
        self.new_optima_energies = [[] for i in range(self.nruns)]  # new minima
        self.new_optima_samples = [[] for i in range(self.nruns)]  # new minima

        # set initial values

        self.E0 = scores[1]  # initialize the 'best score' value
        if self.preset_minimum is not None:
            self.absMin = -self.preset_minimum
        else:
            self.absMin = np.amin(self.E0)
        self.all_scores[0] = scores[1]
        self.all_energies[0] = energy[1]
        self.all_uncertainties[0] = std_dev[1]
        self.all_samples[0] = self.config

        for i in range(self.nruns):
            self.new_optima_samples[i].append(self.config[i])
            self.new_optima_energies[i].append(energy[1][i])
            self.new_optima_scores[i].append(scores[1][i])
            self.new_optima_inds[i].append(0)

    def initRecs(self):
        '''
        step-by-step records for debugging purposes
        :return:
        '''
        self.temprec = [[] for i in range(self.nruns)]
        self.accrec = [[] for i in range(self.nruns)]
        self.stunrec = [[] for i in range(self.nruns)]
        self.scorerec = [[] for i in range(self.nruns)]
        self.enrec = [[] for i in range(self.nruns)]
        self.std_devrec = [[] for i in range(self.nruns)]
        self.vdwrec = [[] for i in range(self.nruns)]

    def initConvergenceStats(self):
        # convergence stats
        self.resetInd = [0 for i in range(self.nruns)]  # flag
        self.acceptanceRate = np.zeros(self.nruns)  # rolling MCMC acceptance rate

    def computeSTUN(self, scores):
        """
        compute the STUN function for the given energies
        :return:
        """
        return 1 - np.exp(-self.gammas * (scores - self.absMin))  # compute STUN function with shared global minimum

    def converge(self, model, builder, crystaldata, iters):
        """
        run the sampler until we converge to an optimum
        :return:
        """
        self.initConvergenceStats()
        self.resampleRandints()

        model = model.cuda()
        crystaldata = crystaldata.cuda()

        self.run_iters = iters
        for self.iter in tqdm.tqdm(range(self.run_iters)):  # sample for a certain number of iterations
            self.iterate(model, builder, crystaldata)  # try a monte-carlo step!

            if (self.iter % self.deltaIter == 0) and (self.iter > 0):  # every N iterations do some reporting / updating
                self.updateAnnealing(crystaldata)  # change temperature or other conditions

            if self.iter % self.randintsResampleAt == 0:  # periodically resample random numbers
                self.resampleRandints()

        print("{} samples were recorded on this run".format(len(np.concatenate(self.all_samples))))

    def prop_configs(self, ind):
        """
        propose a new ensemble of configurations
        :param ind:
        :return:
        """
        self.prop_config = np.copy(self.config)
        for i in range(self.nruns):
            self.prop_config[i, self.pickDimRandints[i, ind]] = self.move_randns[i, ind]

    def iterate(self, model, builder, crystaldata):
        """
        run chainLength cycles of the sampler
        process: 1) propose state, 2) compute acceptance ratio, 3) sample against this ratio and accept/reject move
        :return: config, energy, and stun function will update
        """
        self.ind = self.iter % self.randintsResampleAt  # random number index

        # propose a new state
        self.prop_configs(self.ind)

        # even if it didn't change, just run it anyway (big parallel - to hard to disentangle)
        # compute acceptance ratio
        self.scores, self.energy, self.std_dev, self.vdw_penalty = self.getScores(self.prop_config, self.config, model, builder, crystaldata)

        if self.iter == 0:  # initialize optima recording
            self.initOptima(self.scores, self.energy, self.std_dev)

        self.F, self.DE = self.getDelta(self.scores)
        self.acceptanceRatio = np.minimum(1, np.exp(-self.DE / self.temperature))
        self.updateConfigs()

    def updateConfigs(self):
        '''
        check Metropolis conditions, update configurations, and record statistics
        :return:
        '''
        # accept or reject
        for i in range(self.nruns):
            if self.alphaRandoms[i, self.ind] < self.acceptanceRatio[i]:  # accept
                self.config[i] = np.copy(self.prop_config[i])
                self.recInds[i].append(self.iter)

                if (self.scores[0][i] < self.E0[i]):
                    self.updateBest(i)

        self.recordTrajectory()  # if we accept the move, update the trajectory

        if self.debug:  # record a bunch of detailed outputs
            self.recordStats()

    def recordTrajectory(self):
        self.all_scores[self.iter] = self.scores[0]
        self.all_energies[self.iter] = self.energy[0]
        self.all_uncertainties[self.iter] = self.std_dev[0]
        self.all_samples[self.iter] = self.prop_config
        self.all_vdw_penalties[self.iter] = self.vdw_penalty

    def getDelta(self, scores):
        if self.STUN == 1:  # compute score difference using STUN
            F = self.computeSTUN(scores)
            DE = F[1] - F[0]
        else:  # compute raw score difference
            F = [0, 0]
            DE = scores[1] - scores[0]

        return F, DE

    def recordStats(self):
        for i in range(self.nruns):
            self.temprec[i].append(self.temperature[i])
            self.accrec[i].append(self.acceptanceRate[i])
            self.scorerec[i].append(self.scores[0][i])
            self.enrec[i].append(self.energy[0][i])
            # self.std_devrec[i].append(self.std_dev[0][i])
            if self.STUN:
                self.stunrec[i].append(self.F[0][i])
            self.vdwrec[i].append(self.vdw_penalty[i])

    def getScores(self, prop_config, config, model, builder, crystaldata):
        """
        compute score against which we're optimizing
        :param prop_config:
        :param config:
        :return:
        """
        model.eval()
        with torch.no_grad():
            supercells, _, _ = builder.build_supercells(crystaldata.clone(), torch.Tensor(config).cuda(),
                                                        supercell_size=self.supercell_size, graph_convolution_cutoff=self.graph_convolution_cutoff)
            energy = [-softmax_and_score(model(supercells).cpu().detach().numpy())]

            vdw_penalty = vdw_overlap(supercells, self.vdw_radii).cpu().detach().numpy()

            supercells, _, _ = builder.build_supercells(crystaldata.clone(), torch.Tensor(prop_config).cuda(),
                                                        supercell_size=self.supercell_size, graph_convolution_cutoff=self.graph_convolution_cutoff)
            energy.append(-softmax_and_score(model(supercells).cpu().detach().numpy()))

        std_dev = [np.ones_like(energy[0]), np.ones_like(energy[0])]  # not using this
        score = energy

        return score, energy, std_dev, vdw_penalty

    def updateBest(self, ind):
        self.E0[ind] = self.scores[0][ind]
        if self.E0[ind] < self.absMin:
            self.absMin = self.E0[ind]
        # self.new_optima_samples[ind].append(self.prop_config[ind])
        # self.new_optima_energies[ind].append(self.energy[0][ind])
        self.new_optima_scores[ind].append(self.scores[0][ind])
        self.new_optima_inds[ind].append(self.iter)

    def updateAnnealing(self, crystaldata):
        """
        Following "Adaptation in stochatic tunneling global optimization of complex potential energy landscapes"
        1) updates temperature according to STUN threshold to separate "Local search" and "tunneling" phases
        2) determines when the algorithm is no longer efficiently searching and adapts by resetting the config to a random value
        """
        # 1) if rejection rate is too high, switch to tunneling mode, if it is too low, switch to local search mode
        # acceptanceRate = len(self.stunRec)/self.iter # global acceptance rate


        for i in range(self.nruns):
            acceptedRecently = np.sum((self.iter - np.asarray(self.recInds[i][-self.acceptance_history:])) < self.acceptance_history)  # rolling acceptance rate - how many accepted out of the last hundred iters
            self.acceptanceRate[i] = acceptedRecently / self.acceptance_history

            if self.acceptanceRate[i] < self.target_acceptance_rate:
                self.temperature[i] = self.temperature[i] * (1 + np.random.random(1)[0])  # modulate temperature semi-stochastically
            else:
                self.temperature[i] = self.temperature[i] * (1 - np.random.random(1)[0])

            # if we haven't found a new minimum in a long time, randomize input and do a temperature boost
            if (self.iter - self.resetInd[i]) > self.reset_patience:  # within xx of the last reset
                if (self.iter - self.new_optima_inds[i][-1]) > self.new_minimum_patience:  # haven't seen a new near-minimum in xx steps
                    self.resetInd[i] = self.iter
                    self.resetConfig(crystaldata, i)  # re-randomize selected trajectories
                    self.temperature[i] = self.temp0  # boost temperature
