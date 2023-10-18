
import sys
import numpy as np
import utils.utilsgeo as ug
from scipy.spatial import distance


def create_input(argv):

    locate_dataset = sys.argv[1]
    k = sys.argv[2]

    X_train = np.load(locate_dataset + 'X_train.npy', allow_pickle=True)
    X_test = np.load(locate_dataset + 'X_test.npy', allow_pickle=True)
    y_train = np.load(locate_dataset + 'y_train.npy', allow_pickle=True)
    y_test = np.load(locate_dataset + 'y_test.npy', allow_pickle=True)

    idx, dist = ug.create_sequence(X_train, X_test, int(k))
    idx_eucli, dist_eucli = ug.recovery_dist(idx[:X_train.shape[0], :],
                                             idx[X_train.shape[0]:, :], X_train[:, 2:],
                                             X_test[:, 2:], distance.euclidean)

    idx_geo = idx
    dist_geo = dist
    idx_eucli = idx_eucli
    dist_eucli = dist_eucli

    np.savez_compressed(locate_dataset + 'data.npz', dist_eucli=dist_eucli, dist_geo=dist_geo, idx_eucli=idx_eucli,
                        idx_geo=idx_geo, X_test=X_test, X_train=X_train, y_test=y_test, y_train=y_train)

if __name__ == '__main__':
    create_input(sys.argv[1:])
