import numpy as np
from numba import njit

#written by Lucas Tepper (l.tepper@fu-berlin.de)

@njit
def _compute_fp_events(crossing_indexes, crossing_times, fpts, fpts_sqd, counts):
    """ Computes fpt evens from prepared arrays. Expects an array which contains the index of the
    bin the time series crosses into and and the time passes corresponding
    to that crossing event. Updates the result arrays inplace
    (fpts, fpts_sqd, counts). """

    # found = np.ones(len(counts), dtype=np.int64)
    index_to_start = 1
    for i in range(index_to_start, len(crossing_indexes)):
        crossing_to = crossing_indexes[i]
        j = i - 1
        crossing_from = crossing_indexes[j]
        # count how often destination is reached from source, not how many
        # crossings of source happend before reaching destination
        # for k in range(len(found)):
        #     found[k] = 1
        while crossing_from != crossing_to:
            delta_t = crossing_times[i] - crossing_times[j]
            fpts[crossing_from, crossing_to] += delta_t
            fpts_sqd[crossing_from, crossing_to] += delta_t ** 2
            counts[crossing_from, crossing_to] += 1 #* found[crossing_from]
            # found[crossing_from] = 0
            # set for next
            j -= 1
            if j >= 0:
                crossing_from = crossing_indexes[j]
            else:
                crossing_from = crossing_to


def handle_jumps_multiple_bins(indexes, times):
    """ compute_fp_events does not consider all crossings when the time series crosses multiple
    bins at once. As an easy solution, for every time the trajectory jumps over multiple bins
    at once, we split it into jumps of one bin with the same time stamp. """

    crossings = np.where((indexes[1:] - indexes[:-1]) != 0)[0]
    n_bins_crossed = np.abs(indexes[crossings + 1] - indexes[crossings + 1])
    multiple_crossing = np.where(n_bins_crossed > 1)[0]
    if len(multiple_crossing) > 1:
        for i, __ in enumerate(reversed(multiple_crossing)):
            # keep track of how much index i has changed by adding elements
            elems_to_add = abs(indexes[i] - indexes[i + 1]) - 1
            starting_from = indexes[crossings[i]]
            going_to = indexes[crossings[i] + 1]
            indexes = np.insert(indexes, i + 1, range(starting_from + 1, going_to))
            times = np.insert(times, i + 1, [times[i]] * elems_to_add)
    return indexes, times


def calc_mfpt(traj, start, end, n_points, dt):
    """ Compute first passage events for a given trajectory. Consider all possiple passage
    events over regularly interspaced points over a given intervall. Returns the mean
    of all first passage events, the second moment and the number of counts.
    Arguments:
    start [float]: start of the interval over which to consider fpts
    end [float]: end of the interval over which to consider fpts
    n_points [int]: number of poits over which to consider fpts.
    dt [float]: time step of traj
    Returns:
    mfpt [np.ndarray (n_points, n_points)] mean first passage times. The array element i, j
    contains the mfpt from
    start + i * (end - start) / (n_points - 1)
    -> start + j * (end - start) / (n_points - 1)
    mfpt_std [np.ndarray (n_points, n_points)] second moment of mfpt
    counts [np.ndarray (n_points, n_points)] number of first passage events recorded. """

    width_bins = (end - start) / (n_points - 1)
    fpts = np.zeros((n_points, n_points))
    fpts_sqd = np.zeros((n_points, n_points))
    counts = np.zeros((n_points, n_points)).astype(np.int64)
    # map trajectory to index based on which point it is between, or over or under the range
    indexes = ((traj - start) // width_bins).astype(np.int64)
    indexes = np.maximum(indexes, -1)
    indexes = np.minimum(indexes, n_points - 1)
    times = (np.arange(len(traj)) + 1) * dt
    # add fictions crossing event for multiple jumps
    # indexes [2, 4]; times [0.1, 0.2] -> [2, 3, 4]; [0.1, 0.2, 0.2]
    indexes, times = handle_jumps_multiple_bins(indexes, times)
    # find the crossing events
    crossings = np.where((indexes[1:] - indexes[:-1]) != 0)[0]
    # and into which bin the transition happens
    crossing_indexes = (indexes[crossings] + indexes[crossings + 1]) // 2 + 1
    # and the corresponding times
    crossing_times = times[crossings]
    _compute_fp_events(crossing_indexes, crossing_times, fpts, fpts_sqd, counts)
    # get results
    mask = counts != 0
    mfpt = fpts
    mfpt[mask] /= counts[mask]
    mfpt_sqd = fpts_sqd
    mfpt_sqd[mask] /= counts[mask]
    mfpt_std = mfpt_sqd - mfpt ** 2
    mfpt_std[mask] = np.sqrt(mfpt_std[mask])
    return mfpt, mfpt_std, counts