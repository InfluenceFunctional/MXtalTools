# def sampling_prep(self):
#     dataset_builder = self.misc_pre_training_items()
#     del dataset_builder
#     generator, discriminator, generator_optimizer, generator_schedulers, \
#         discriminator_optimizer, discriminator_schedulers, params1, params2 \
#         = self.init_models()
#
#     self.config.current_batch_size = self.config.min_batch_size
#
#     extra_test_set_path = self.config.extra_test_set_paths
#     extra_test_loader = get_extra_test_loader(self.config, extra_test_set_path, dataDims=self.config.dataDims,
#                                               pg_dict=self.point_groups, sg_dict=self.space_groups,
#                                               lattice_dict=self.lattice_type, sym_ops_dict=self.sym_ops)
#
#     self.randn_generator = independent_gaussian_model(input_dim=self.config.dataDims['num lattice features'],
#                                                       means=self.config.dataDims['lattice_means'],
#                                                       stds=self.config.dataDims['lattice_stds'],
#                                                       normed_length_means=self.config.dataDims[
#                                                           'lattice normed length means'],
#                                                       normed_length_stds=self.config.dataDims[
#                                                           'lattice normed length stds'],
#                                                       cov_mat=self.config.dataDims['lattice cov mat'])
#
#     # blind_test_identifiers = [
#     #     'OBEQUJ', 'OBEQOD','NACJAF'] # targets XVI, XVII, XXII
#     #
#     # single_mol_data = extra_test_loader.dataset[extra_test_loader.csd_identifier.index(blind_test_identifiers[-1])]
#
#     return extra_test_loader, generator, discriminator, \
#         generator_optimizer, generator_schedulers, discriminator_optimizer, discriminator_schedulers, \
#         params1, params2
#
# # def model_sampling(self):  # todo combine with mini-CSP module
# #     """ DEPRECATED
# #     Stun MC annealing on a pretrained discriminator / generator
# #     """
# #     with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username,
# #                     tags=[self.config.wandb.experiment_tag]):
# #         extra_test_loader, generator, discriminator, \
# #             generator_optimizer, generator_schedulers, discriminator_optimizer, discriminator_schedulers, \
# #             params1, params2 = self.sampling_prep()  # todo rebuild this with new model_init
# #
# #         generator.eval()
# #         discriminator.eval()
# #
# #         smc_sampler = mcmcSampler(
# #             gammas=np.logspace(-4, 0, self.config.current_batch_size),
# #             seedInd=0,
# #             STUN_mode=False,
# #             debug=False,
# #             init_adaptive_step_size=self.config.sample_move_size,
# #             global_temperature=0.00001,  # essentially only allow downward moves
# #             generator=generator,
# #             supercell_size=self.config.supercell_size,
# #             graph_convolution_cutoff=self.config.discriminator.graph_convolution_cutoff,
# #             vdw_radii=self.vdw_radii,
# #             preset_minimum=None,
# #             spacegroup_to_search='P-1',  # self.config.generate_sgs,
# #             new_minimum_patience=25,
# #             reset_patience=50,
# #             conformer_orientation=self.config.generator.canonical_conformer_orientation,
# #         )
# #
# #         '''
# #         run sampling
# #         '''
# #         # prep the conformers
# #         single_mol_data_0 = extra_test_loader.dataset[0]
# #         collater = Collater(None, None)
# #         single_mol_data = collater([single_mol_data_0 for n in range(self.config.current_batch_size)])
# #         single_mol_data = self.set_molecule_alignment(single_mol_data,
# #                                                       mode_override='random')  # take all the same conformers for one run
# #         override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index('P-1') + 1
# #         sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(
# #             single_mol_data.x.device) for i in range(single_mol_data.num_graphs)]
# #         single_mol_data = write_sg_to_all_crystals('P-1', self.supercell_builder.dataDims, single_mol_data,
# #                                                    self.supercell_builder.symmetries_dict, sym_ops_list)
# #
# #         smc_sampling_dict = smc_sampler(discriminator, self.supercell_builder,
# #                                         single_mol_data.clone().to(self.config.device), None,
# #                                         self.config.sample_steps)
# #
# #         '''
# #         reporting
# #         '''
# #
# #         sampling_telemetry_plot(self.config, wandb, smc_sampling_dict)
# #         cell_params_tracking_plot(wandb, self.supercell_builder, self.layout, self.config, smc_sampling_dict, collater, extra_test_loader)
# #         best_smc_samples, best_smc_samples_scores, best_smc_cells = sample_clustering(self.supercell_builder, self.config, smc_sampling_dict,
# #                                                                                       collater,
# #                                                                                       extra_test_loader,
# #                                                                                       discriminator)
# #         # destandardize samples
# #         unclean_best_samples = de_clean_samples(self.supercell_builder, best_smc_samples, best_smc_cells.sg_ind)
# #         single_mol_data = collater([single_mol_data_0 for n in range(len(best_smc_samples))])
# #         gd_sampling_dict = gradient_descent_sampling(
# #             discriminator, unclean_best_samples, single_mol_data.clone(), self.supercell_builder,
# #             n_iter=500, lr=1e-3,
# #             optimizer_func=optim.Rprop,
# #             return_vdw=True, vdw_radii=self.vdw_radii,
# #             supercell_size=self.config.supercell_size,
# #             cutoff=self.config.discriminator.graph_convolution_cutoff,
# #             generate_sgs='P-1',  # self.config.generate_sgs
# #             align_molecules=True,  # always true here
# #         )
# #         gd_sampling_dict['canonical samples'] = gd_sampling_dict['samples']
# #         gd_sampling_dict['resampled state record'] = [[0] for _ in range(len(unclean_best_samples))]
# #         gd_sampling_dict['scores'] = gd_sampling_dict['scores'].T
# #         gd_sampling_dict['vdw penalties'] = gd_sampling_dict['vdw'].T
# #         gd_sampling_dict['canonical samples'] = np.swapaxes(gd_sampling_dict['canonical samples'], 0, 2)
# #         sampling_telemetry_plot(self.config, wandb, gd_sampling_dict)
# #         cell_params_tracking_plot(wandb, self.supercell_builder, self.layout, self.config, gd_sampling_dict, collater, extra_test_loader)
# #         best_gd_samples, best_gd_samples_scores, best_gd_cells = sample_clustering(self.supercell_builder, self.config, gd_sampling_dict, collater,
# #                                                                                    extra_test_loader,
# #                                                                                    discriminator)
# #
# #         # todo process refined samples
# #         # todo compare final samples to known minima
# #
# #         extra_test_sample = next(iter(extra_test_loader)).cuda()
# #         sample_supercells = self.supercell_builder.unit_cell_to_supercell(extra_test_sample, self.config.supercell_size, self.config.discriminator.graph_convolution_cutoff)
# #         known_sample_scores = softmax_and_score(discriminator(sample_supercells.clone()))
# #
# #         aa = 1
# #         plt.clf()
# #         plt.plot(gd_sampling_dict['scores'])
# #
# #         # np.save(f'../sampling_output_run_{self.config.run_num}', sampling_dict)
# #         # self.report_sampling(sampling_dict)
