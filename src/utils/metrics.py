import numpy as np

def compute_rt_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    r1 = float(np.sum(ind == 0))  / len(ind)
    r5 = float(np.sum(ind < 5))  / len(ind)
    r10 = float(np.sum(ind < 10))  / len(ind)
    r50 = float(np.sum(ind < 50))  / len(ind)
    medr = np.median(ind) + 1
    meanr  = np.mean(ind) + 1
    return r1, r5, r10, r50, medr, meanr


if __name__ == '__main__':

    sim_matrix = np.random.random((5,5))
    print(sim_matrix)
    print(t2v_metric(sim_matrix))
    print(sim_matrix.T)
    print(v2t_metric(sim_matrix))


