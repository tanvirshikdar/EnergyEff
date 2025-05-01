import mxnet as mx
import math

def prod(shape):
    """
    Compute the product of dimensions of a shape (e.g., for calculating memory usage).

    Parameters
    ----------
    shape : tuple
        A tuple representing the shape (dimensions) of the data.

    Returns
    -------
    int
        The product of all the dimensions.
    """
    return math.prod(shape)


def is_param(name):
    """
    Check if the given symbol name corresponds to a parameter (weights, bias, etc.).

    Parameters
    ----------
    name : str
        The name of the symbol.

    Returns
    -------
    bool
        True if the name corresponds to a parameter, False otherwise.
    """
    return name != 'data' and (name.endswith('weight') or name.endswith('bias') or 
                               name.endswith('gamma') or name.endswith('beta'))


def make_mirror_plan(sym, threshold, plan_info=None, **kwargs):
    """
    Create a memory allocation plan based on the given threshold for optimizing memory usage.

    The function analyzes the symbolic network and adjusts memory allocation by applying
    the `force_mirroring` attribute to appropriate nodes, optimizing memory usage.

    Parameters
    ----------
    sym : mxnet.symbol
        The symbolic network to optimize.
    threshold : int
        The threshold for memory usage, used to determine the point where computations
        can be recomputed instead of being stored.
    plan_info : dict, optional
        A dictionary to store information about the memory plan. Defaults to None.
    **kwargs : dict
        Additional keyword arguments to pass to `infer_shape`.

    Returns
    -------
    mxnet.symbol
        The optimized symbolic network with memory allocations adjusted.
    """
    threshold <<= 20  # Convert threshold to bytes (from MB)
    sym_copy = sym.__copy__()  # Create a copy of the original symbol
    internals = sym_copy.get_internals()
    _, out_shapes, _ = internals.infer_shape(**kwargs)
    shape_dict = list(zip(internals.list_outputs(), out_shapes))

    total_size, param_size, local_size, save_size, max_size = 0, 0, 0, 0, 0
    last_stage = ''
    stage_decision = ''

    # Loop through the network and apply memory optimizations
    for idx, (name, shape) in enumerate(shape_dict):
        sb = internals[idx]

        if is_param(name):
            param_size += prod(shape) * 4  # Accumulate parameter memory size
            continue
        else:
            total_size += prod(shape) * 4
            local_size += prod(shape) * 4
            sb._set_attr(force_mirroring='True')  # Mark the symbol as recomputable

        # Mirror stage logic
        if sb.attr('mirror_stage') is not None:
            stage = sb.attr('mirror_stage')
            if stage != last_stage:
                if local_size > threshold:
                    save_size += prod(shape) * 4
                    max_size = max(max_size, local_size)
                    local_size = 0
                    stage_decision = 'False'
                    sb._set_attr(force_mirroring=stage_decision)
                else:
                    stage_decision = 'True'
                last_stage = stage
            elif stage == last_stage and stage_decision == 'False':
                save_size += prod(shape) * 4
                sb._set_attr(force_mirroring=stage_decision)

    # Store memory optimization statistics if needed
    if plan_info is not None:
        plan_info['max_size'] = max_size
        plan_info['save_size'] = save_size

    return sym_copy


def get_cost(sym, type_dict=None, **kwargs):
    """
    Compute the memory cost of the symbolic network by running a bind operation on the CPU.

    Parameters
    ----------
    sym : mxnet.symbol
        The symbolic network.
    type_dict : dict, optional
        The type dictionary for the symbols. Defaults to None.
    **kwargs : dict
        Additional arguments passed to `simple_bind`.

    Returns
    -------
    int
        The memory cost (in bytes).
    """
    texec = sym.simple_bind(ctx=mx.gpu(),
                            grad_req='write',
                            type_dict=type_dict,
                            **kwargs)
    # Extract and return the memory usage from the debug string
    return int(texec.debug_str().split('\n')[-3].split()[1])


def search_plan(sym, ntrial=6, type_dict=None, **kwargs):
    """
    Heuristic search to find the best memory allocation plan.

    The function tries different memory thresholds and compares the costs, 
    iterating over the possible memory plans to find the one with the lowest cost.

    Parameters
    ----------
    sym : mxnet.symbol
        The symbolic configuration of the network.
    ntrial : int, optional
        The number of additional grid search steps (default is 6).
    type_dict : dict, optional
        The type dictionary for the symbols (default is None).
    **kwargs : dict
        Additional arguments passed to `simple_bind` and `make_mirror_plan`.

    Returns
    -------
    mxnet.symbol
        The optimized network with the best memory allocation plan.
    """
    history = []
    threshold = 0
    min_threshold, min_cost = None, None
    nbegin = 3

    for k in range(nbegin):
        info = {}
        # Apply the memory optimization plan
        sym = make_mirror_plan(sym, threshold=threshold, plan_info=info, **kwargs)
        cost = get_cost(sym, type_dict, **kwargs)
        save_size = info['save_size'] >> 20  # Convert to MB
        local_size = info['max_size'] >> 20  # Convert to MB
        guess = int(math.sqrt(save_size * local_size / 2))

        if min_cost is None or min_cost > cost:
            min_cost = cost
        if min_threshold is None or local_size < min_threshold:
            min_threshold = local_size

        print(f"Search threshold={threshold} MB, cost={cost} MB")
        history.append((cost, threshold, sym))
        threshold = guess

    max_threshold = threshold * math.sqrt(2)
    step = int((max_threshold - min_threshold) / ntrial)
    threshold = min_threshold + step

    # Search for the best plan with refined thresholds
    if step > 0:
        for k in range(ntrial):
            sym = make_mirror_plan(sym, threshold=threshold, plan_info=info, **kwargs)
            cost = get_cost(sym, type_dict, **kwargs)
            print(f"Search threshold={threshold} MB, cost={cost} MB")
            history.append((cost, threshold, sym))
            threshold += step

    # Sort the history and return the plan with the lowest cost
    history.sort(key=lambda x: x[0])
    cost, threshold, sym = history[0]
    print(f"Found best plan with threshold={threshold} MB, cost={cost} MB")
    return sym