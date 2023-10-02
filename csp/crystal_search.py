from torch_geometric.loader.dataloader import Collater
import torch
import numpy as np
from tqdm import tqdm

class CSP:
    def __init__(self, device, config, dataDims, std_dict,
                 supercell_builder, generator, discriminator):
        self.device = device
        self.dataDims = dataDims
        self.std_dict = std_dict
        self.generator = generator.eval()
        self.discriminator = discriminator.eval()
        self.supercell_builder = supercell_builder

        self.sgs_to_search = config.sgs_to_search  # todo make these args
        self.num_samples = config.num_samples

    def search(self, crystaldata, batch_size):
        """
        perform a CSP search for a single crystal target, crystaldata
        """
        '''prepare data'''
        collater = Collater(None, None)
        crystaldata_batch = collater([crystaldata for _ in range(batch_size)]).to(self.device)
        
        self.generate_samples(crystaldata_batch, batch_size)


    def generate_samples(self, crystaldata_batch, batch_size):
        num_iters = np.ceil(self.num_samples // batch_size)

        with torch.no_grad():
            for ii in tqdm(range(num_iters)):
                generator_data = crystaldata_batch.clone().to(self.device)

                # use generator to make samples
                samples, prior, standardized_target_packing_coeff, fake_data = \
                    self.get_generator_samples(fake_data)

                generated_supercells, generated_cell_volumes = \
                    self.supercell_builder.build_supercells(
                        fake_data, samples, self.config.supercell_size,
                        self.config.discriminator.model.convolution_cutoff,
                        align_to_standardized_orientation=False,
                    )