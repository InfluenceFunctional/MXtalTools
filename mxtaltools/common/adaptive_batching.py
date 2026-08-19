import gc
import time
import warnings
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.dataset_utils.utils import collate_data_list

#: Warn once per process, not once per chunk -- a bulk scan calls this hundreds of
#: times and a repeated warning is noise nobody reads.
_GRAD_WARNED = False

_GRAD_WARNING = (
    'adaptive_batched_analysis called with gradients ENABLED. This helper is only '
    'ever used for offline bulk scoring, and fairchem leaves grad on by design '
    '(its _run_inference picks nullcontext whenever direct_forces is False, which '
    'the crystal predictor sets), so every scored row comes back carrying grad_fn '
    'and pins its activations -- measured at 100-250 MB PER CRYSTAL. The symptom is '
    'not a crash: the OOM handler below shrinks the batch and makes that size '
    'sticky, so the scan merely runs slower and slower. If you are not '
    'differentiating these energies, wrap the call in torch.no_grad().')


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
    # WARN, DO NOT SILENTLY FORCE no_grad. Overriding grad semantics inside a shared
    # helper is how a legitimate gradient path would start reading zero with nothing
    # to show for it. The caller owns the decision; this only makes the cost visible.
    global _GRAD_WARNED
    if torch.is_grad_enabled() and not _GRAD_WARNED:
        warnings.warn(_GRAD_WARNING, RuntimeWarning, stacklevel=2)
        _GRAD_WARNED = True

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
                    # name the likeliest cause in the failure itself: one crystal
                    # does not exhaust a card, but a few hundred retained autograd
                    # graphs do, and that reads as "this crystal is too big"
                    hint = ('  GRAD WAS ENABLED -- retained autograd graphs are the '
                            'usual cause; see the warning at call entry.'
                            if torch.is_grad_enabled() else '')
                    raise RuntimeError(
                        f"Cascading OOM failure: batch_size already 1.{hint}"
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

    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    pbar.close()
    if return_state:
        return collate_data_list(outputs_list), state
    else:
        return collate_data_list(outputs_list)
