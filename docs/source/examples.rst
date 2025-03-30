Examples
================

.. default-role:: math

MXtalTools has several analysis utilities for various molecular crystal tasks.

Crystal Density Prediction
------------------------------
Our model is trained on CSD data to predict the crystal packing coefficient, `c_{p = \frac{V_{mol}}{V_{aunit}}`, with `V_{mol}` the molecule volume, as estimated by our built-in utility, and `V_{aunit}` the volume of a crystal asymmetric unit, `V_{aunit}=\frac{V_{cell}}{Z}`.

While we achieve good accuracy and have some ability to estimate the prediction error via dropout variance, the model may not perform well for molecules unlike those in the CSD.

For a runnable example, see examples/crystal_density_prediction.py.

First we load some molecules.
Here we are having RDKit build and minimize molecules generated from some SMILES codes.
This is automated in our :class:`MolData` class.
One can also directly input the atom types and coordinates.

.. code-block:: python

        device = 'cpu'
        base_molData = MolData()
        num_mols = len(test_smiles)
        mols = [base_molData.from_smiles(test_smiles[ind],
                                         compute_partial_charges=True,
                                         minimize=True,
                                         protonate=True,
                                         ) for ind in range(num_mols)]
        mols = [mol for mol in mols if mol is not None]  # sometimes the embedding fails
        mol_batch = collate_data_list(mols).to(device)

Then, we load the pre-trained model:

.. code-block:: python

        model = load_molecule_scalar_regressor(checkpoint, device)

Now, we make predictions using the model, which we can easily convert from packing coefficient, to `V_{aunit}`, to the density.

.. code-block:: python

        packing_coeff_pred = model(mol_batch).flatten() * model.target_std + model.target_mean
        aunit_volume_pred = mol_batch.mol_volume / packing_coeff_pred  # A^3
        density_pred = mol_batch.mass / aunit_volume_pred * 1.6654  # g/cm^3

Alternatively, to estimate uncertainty, we resample using dropout:

.. code-block:: python

        predictions = []
        num_samples = 50
        model = enable_dropout(model)
        for _ in range(num_samples):
            predictions.append(model(mol_batch).flatten() * model.target_std + model.target_mean)

        predictions = torch.stack(predictions)
        packing_coeff_mean = predictions.mean(0)
        packing_coeff_std = predictions.std(0)


Molecule Encoding
------------------------------

This module uses a pretrained Mo3ENet to convert a molecule point cloud into equivariant vector and scalar representations.
To-date, Mo3ENet has been trained on QM9-like molecules, that is, molecules with 9-or-less heavy atoms and containing H, C, N, O, and F.
We tend to observe poorer performance on fluorine-rich or highly symmetric molecules.
We include in this example a utility to check the fidelity of each molecule representation.

For a runnable example, see examples/molecule_autoencoder.py.

First we load some molecules, and ensure they are each centered on the origin.
Here we are having RDKit build and minimize molecules generated from some SMILES codes.
This is automated in our MolData class.
One can also directly input the atom types and coordinates.

.. code-block:: python

        device = 'cpu'
        base_molData = MolData()
        num_mols = len(test_smiles)
        mols = [base_molData.from_smiles(test_smiles[ind],
                                         compute_partial_charges=True,
                                         minimize=True,
                                         protonate=True,
                                         ) for ind in range(num_mols)]
        mols = [mol for mol in mols if mol is not None]  # sometimes the embedding fails
        mol_batch = collate_data_list(mols).to(device)
        mol_batch.recenter_molecules()

Then, we load the pre-trained model

.. code-block:: python

        model = load_molecule_autoencoder(checkpoint,device)

Get the vector and scalar embeddings

.. code-block:: python

        vector_encoding = model.encode(mol_batch.clone())
        scalar_encoding = model.scalarizer(vector_encoding)

and optionally decode the embedding and visually inspect the reconstruction to ensure the quality of the reqpresentation is sufficienlty high

.. code-block:: python
        reconstruction_loss, rmsd, matched_molecule = model.check_embedding_quality(mol_batch, visualize=True)

Crystal Analysis & Scoring
------------------------------

This module combines many utilities for the construction and analysis of molecular crystal data, including a score model trained on CSD data and applicable to molecules within the "CSD distribution".

For a runnable example, see examples/crystal_analysis.py.

We start by initializing some configs

.. code-block:: python
        device = 'cpu'
        mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
        checkpoint = r"../models/crystal_discriminator.pt"
        space_groups_to_sample = ["P1", "P-1", "P21/c", "C2/c", "P212121"]
        sym_info = init_sym_info()

We can then load some example crystals, included in the MXtalTools package, and generate a batch.

.. code-block:: python

        example_crystals = torch.load(mini_dataset_path)
        crystal_batch = collate_data_list(example_crystals[:10])

A core function of our code is crystal parameterization and construction, and so we show a simple example of building crystals, starting from the same molecules as before, but with random space groups and lattice parameters.

.. code-block::
        # initialize prior distribution
        crystal_batch2 = crystal_batch.detach().clone()
        prior = CSDPrior(sym_info=sym_info, device=device)
        # pick space groups to sample
        sgs_to_build = np.random.choice(space_groups_to_sample, replace=True, size=crystal_batch.num_graphs)
        sg_rand_inds = torch.tensor([list(sym_info['space_groups'].values()).index(SG) + 1 for SG in sgs_to_build],
                                    dtype=torch.long, device=device)  # indexing from 0
        crystal_batch2.reset_sg_info(sg_rand_inds)
        # sample cell parameters
        normed_cell_params = prior(len(crystal_batch), sg_rand_inds).to(crystal_batch.device)
        # assign new parameters to crystal
        normed_cell_lengths, crystal_batch2.cell_angles, crystal_batch2.aunit_centroid, crystal_batch2.aunit_orientation = normed_cell_params.split(
            3, dim=1)
        crystal_batch2.cell_lengths = crystal_batch2.denorm_cell_lengths(normed_cell_lengths)
        crystal_batch2.box_analysis()

We can then load the crystal scoring model

.. code-block:: python

        model = load_crystal_score_model(
            checkpoint,
            device
        )

And proceed to analyzing both sets of crystals.
We present here a very basic analysis, computing a very basic Lennard-Jones-type and short-range electrostatic potential.
We also show the outpudts of the crystal scoring model, (1) it's classification confidence between "real" CSD samples and "fake" samples, not from the CSD, and (2) the predicted distance in RDF space from the given crystal to the "correct" crystal for the given molecule.

.. code-block:: python
        lj_pot, es_pot, scaled_lj_pot, crystal_cluster = crystal_batch.build_and_analyze(return_cluster=True)
        model_output = model(crystal_cluster)
        model_score = softmax_and_score(model_output[:, :2])
        rdf_dist_pred = F.softplus(model_output[:, 2])

        lj_pot2, es_pot2, scaled_lj_pot2, crystal_cluster2 = crystal_batch2.build_and_analyze(return_cluster=True)
        model_output2 = model(crystal_cluster2)
        model_score2 = softmax_and_score(model_output2[:, :2])
        rdf_dist_pred2 = F.softplus(model_output2[:, 2])
