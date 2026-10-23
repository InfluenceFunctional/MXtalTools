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
    Run ``analyze(analyses, assign_outputs=True, **kwargs)`` over a batch in chunks whose size adapts to CUDA
    out-of-memory errors.

    The batch is split into samples with ``batch.batch_to_list()``, which works on a clone, so ``batch`` itself is not
    modified. Each chunk of samples is collated with ``collate_data_list``, moved to ``device``, analysed, moved to CPU
    and split back into samples; the returned batch is collated from all analysed samples.

    Parameters
    ----------
    batch : collated data object
        Provides ``batch_to_list()``. Its samples must collate with ``collate_data_list`` into objects providing
        ``to()``, ``cpu()``, ``batch_to_list()`` and ``analyze()``.
    analyses : list or str
        Analysis names, passed as the first argument of ``analyze``; a str is wrapped in a one-element list.
    state : dict
        Caller-owned dict, mutated in place. ``state['batch_size']`` is reset to ``initial_batch_size`` at entry on
        every call, so no chunk size carries over between calls; it is then updated as the chunk size grows or
        shrinks, and its final value is left in ``state``.
    initial_batch_size : int, optional
        Chunk size at the start of the call. Default 1000.
    max_batch_size : int, optional
        Growth is applied only while the chunk size is at most this value, so the size can exceed it by one growth
        step. Default 100_000.
    grow_factor : float, optional
        After each successful chunk, if no OOM has occurred during this call, the chunk size is at most
        ``max_batch_size`` and is below the number of samples, it grows by ``max(int(batch_size * grow_factor), 1)``.
        Default 0.01.
    shrink_factor : float, optional
        On a CUDA OOM the chunk size becomes ``max(int(batch_size * shrink_factor), 1)`` and the chunk is retried from
        the same sample; the chunk size then no longer grows for the rest of the call. Default 0.65.
    oom_sleep : float, optional
        Seconds slept after an OOM, following ``gc.collect()``, ``torch.cuda.empty_cache()`` and
        ``torch.cuda.synchronize()``. Default 0.1.
    return_state : bool, optional
        If True, also return ``state``. Default False.
    device : str or torch.device, optional
        Device each chunk is moved to before ``analyze``. Default ``'cuda'``.
    show_tqdm : bool, optional
        Show a tqdm progress bar over samples. Default False.
    **kwargs
        Forwarded to ``analyze``.

    Returns
    -------
    out_batch : collated data object
        On CPU, samples in input order, with each analysis output attached as a graph attribute by
        ``analyze(..., assign_outputs=True)``. Attributes that ``collate_data_list`` excludes by default (including
        ``unit_cell_pos``) are not carried through.
    state : dict
        Only when ``return_state`` is True: the dict that was passed in.

    Raises
    ------
    RuntimeError
        On a CUDA OOM when the chunk size is already 1, chained from the original error. A ``RuntimeError`` or
        ``ValueError`` that ``is_cuda_oom`` does not classify as OOM, and any other exception, propagates unchanged.

    Warns
    -----
    RuntimeWarning
        Once per process, if called while gradients are enabled.
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
