import gc
import time
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.dataset_utils.utils import collate_data_list


def adaptive_batched_analysis(
        batch,
        analyses: Union[list, str],
        state: dict,
        *,
        initial_batch_size: int = 1000,
        max_batch_size: int = 100_000,
        grow_factor: float = 0.01,
        shrink_factor: float = 0.65,
        oom_sleep: float = 0.1,
        return_state: bool = False,
        device = 'cuda',
        show_tqdm: bool = False,
        **kwargs,
):
    """
    Run batch.analyze(analysis_name, assign_outputs=True, **kwargs) over the
    full batch using adaptive mini-batches to handle GPU OOM gracefully.

    Parameters
    ----------
    batch           Any batch object with .batch_to_list() and .analyze().
    analysis_name   Passed as the first argument to batch.analyze().
    state           Mutable dict owned by the caller; used to carry batch_size
                    across retries within a single call. Pass a fresh {} each
                    call if you don't want persistence across calls.
    **kwargs        Forwarded to batch.analyze() (e.g. predictor, temperature).

    Returns
    -------
    Collated batch object with outputs assigned.
    """
    if not hasattr(state, 'batch_size'):
        state["batch_size"] = initial_batch_size

    if isinstance(analyses, str):
        analyses = [analyses]
    data_list = batch.batch_to_list()
    n_samples = len(data_list)
    outputs_list = [None] * n_samples
    cursor = 0
    already_oomed = False
    pbar = tqdm(total=len(data_list), disable=not show_tqdm)

    while cursor < n_samples:
        inds = np.arange(cursor, min(n_samples, cursor + state["batch_size"]))
        sub_batch = collate_data_list([data_list[i] for i in inds])
        sub_batch = sub_batch.to(device)
        try:
            sub_batch.analyze(analyses, assign_outputs=True, **kwargs)
            outputs_list[cursor: cursor + len(inds)] = sub_batch.cpu().batch_to_list()

            cursor += len(inds)
            pbar.update(len(inds))
            if (
                    state["batch_size"] <= max_batch_size
                    and state["batch_size"] < n_samples
                    and not already_oomed
            ):
                state["batch_size"] += max(int(state["batch_size"] * grow_factor), 1)

        except (RuntimeError, ValueError) as e:
            if is_cuda_oom(e):
                if state["batch_size"] == 1:
                    raise RuntimeError(
                        "Cascading OOM failure: batch_size already 1"
                    ) from e
                state["batch_size"] = max(int(state["batch_size"] * shrink_factor), 1)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                already_oomed = True
                time.sleep(oom_sleep)
                # retry same cursor
            else:
                raise

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    pbar.close()
    if return_state:
        return collate_data_list(outputs_list), state
    else:
        return collate_data_list(outputs_list)
