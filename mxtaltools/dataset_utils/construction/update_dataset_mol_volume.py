import torch
from tqdm import tqdm
from mxtaltools.dataset_utils.utils import collate_data_list


def compute_volumes(batch_size, volumes, dataset):
    num_batches = len(dataset) // batch_size
    if len(dataset) % batch_size != 0:
        num_batches += 1
    with torch.no_grad():
        print(f"starting dataset {dataset_path}")
        for batch_ind in tqdm(range(num_batches)):
            start = batch_ind * batch_size
            end = (batch_ind + 1) * batch_size
            batch = collate_data_list(dataset[start:end]).to(device)
            volumes[start:end] = batch.volume_calculation().cpu()

    return volumes


if __name__ == "__main__":
    dataset_paths = [
        r"D:\crystal_datasets\mini_CSD_dataset.pt",
        r"D:\crystal_datasets\test_CSD_dataset.pt",
        r"D:\crystal_datasets\CSD_dataset.pt",
        r"D:\crystal_datasets\mini_qm9_dataset.pt",
        r"D:\crystal_datasets\test_qm9_dataset.pt",
        r"D:\crystal_datasets\qm9_dataset.pt",
        r"D:\crystal_datasets\mini_qm9s_dataset.pt",
        r"D:\crystal_datasets\test_qm9s_dataset.pt",
        r"D:\crystal_datasets\qm9s_dataset.pt",
    ]
    device = 'cuda'
    batch_size = 2000
    for dataset_path in dataset_paths:
        dataset = torch.load(dataset_path, weights_only=False)
        volumes = torch.zeros(len(dataset), dtype=torch.float32, device='cpu')
        finished = False
        while not finished:
            try:
                volumes = compute_volumes(batch_size, volumes, dataset)
                finished = True
            except torch.cuda.OutOfMemoryError as e:  # if we do hit OOM, slash the batch size
                batch_size = int(batch_size * 0.9)
                print(f"batch size slached to {batch_size}")
                torch.cuda.empty_cache()

        for ind in tqdm(range(len(dataset))):
            dataset[ind].mol_volume = volumes[ind]
            if hasattr(dataset[0], 'packing_coeff'):
                dataset[ind].packing_coeff = dataset[ind].mol_volume * dataset[ind].sym_mult / dataset[ind].cell_volume

        torch.save(dataset, dataset_path)
        print('Saved dataset at {}'.format(dataset_path))
