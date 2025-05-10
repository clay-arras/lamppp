import numpy as np

# MATRIX_DIMS = (6, 6)
# MATRIX_DIMS = (12, 12)
# MATRIX_DIMS = (16, 16)
MATRIX_DIMS = (18, 18)


def sample_from_intervals(intervals, shape):
    m = len(intervals)
    n = np.prod(shape)

    def interval_size(interval):
        assert interval[1] >= interval[0]
        return interval[1] - interval[0]

    tot_sz = sum([interval_size(i) for i in intervals])
    p = [interval_size(i) / tot_sz for i in intervals]

    interval_idx = np.random.choice(m, size=n, p=p)
    out = np.empty(n)

    for i, (lo, hi) in enumerate(intervals):
        mask = interval_idx == i
        out[mask] = np.random.uniform(lo, hi, mask.sum())
    return out.reshape(shape).tolist()


def sample_matrices(n, ranges):
    return [sample_from_intervals(ranges, MATRIX_DIMS) for _ in range(n)]
