
import numpy as np
import math
#import geopy.distance
from scipy.spatial import distance
from math import radians, cos, sin, asin, sqrt


#########################################################################################
# metrics calculation
#########################################################################################

# based in https://stackoverflow.com/a/13849249/71522

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def vector_calc(lat, long, ht):
    '''
    Calculates the vector from a specified point on the Earth's surface to the North Pole.
    '''
    a = 6378137.0  # Equatorial radius of the Earth
    b = 6356752.314245  # Polar radius of the Earth

    e_squared = 1 - ((b ** 2) / (a ** 2))  # e is the eccentricity of the Earth
    n_phi = a / (np.sqrt(1 - (e_squared * (np.sin(lat) ** 2))))

    x = (n_phi + ht) * np.cos(lat) * np.cos(long)
    y = (n_phi + ht) * np.cos(lat) * np.sin(long)
    z = ((((b ** 2) / (a ** 2)) * n_phi) + ht) * np.sin(lat)

    x_npole = 0.0
    y_npole = 6378137.0
    z_npole = 0.0

    v = ((x_npole - x), (y_npole - y), (z_npole - z))

    return v


def angle_calc(lat1, long1, lat2, long2, ht1=0, ht2=0):
    '''
    Calculates the angle between the vectors from 2 points to the North Pole.
    '''
    # Convert from degrees to radians
    lat1_rad = (lat1 / 180) * np.pi
    long1_rad = (long1 / 180) * np.pi
    lat2_rad = (lat2 / 180) * np.pi
    long2_rad = (long2 / 180) * np.pi

    v1 = vector_calc(lat1_rad, long1_rad, ht1)
    v2 = vector_calc(lat2_rad, long2_rad, ht2)

    # The angle between two vectors, vect1 and vect2 is given by:
    # arccos[vect1.vect2 / |vect1||vect2|]
    dot = np.dot(v1, v2)  # The dot product of the two vectors
    v1_mag = np.linalg.norm(v1)  # The magnitude of the vector v1
    v2_mag = np.linalg.norm(v2)  # The magnitude of the vector v2

    theta_rad = np.arccos(dot / (v1_mag * v2_mag))
    # Convert radians back to degrees
    theta = (theta_rad / np.pi) * 180

    return theta


# ref https://gist.github.com/jeromer/2005586
def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


# ref https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
# get the Vincenty distance
def dist_geodesic(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km

#########################################################################################
# manage sequences
#########################################################################################


# Create sequence n nearest
def create_sequence(x_train, x_test, seq):
    """
    Creates a sequence of the nearest n points for each point in the dataset.
    :param x_test: np array dataset where lat and long are in first and second position.
    :param x_train: np array dataset where lat and long are in first and second position.
    :param seq: number of the nearest points
    :return: two np arrays. First, the indices of the nearest n points. Second, the distance of the nearest n points.
    """
    # concatenates the training and test set
    dataset = np.concatenate((x_train, x_test), axis=0)
    # the sequence of distances from n neighbors
    nearest_dist = []
    # the sequence of the index from n neighbors
    nearest_idx = []
    # A c x r matrix will be generated. Each column represents the points, and the rows represent the neighbors.
    # The number of columns corresponds to the total of points, while the number of rows corresponds to the number
    # of points in the training set.
    for c in range(dataset.shape[0]):
        dist = []
        for r in range(x_train.shape[0]):
            # To each point c, calculate the distance to each point in the training set.
            dist.append(dist_geodesic(dataset[c, 0], dataset[c, 1], x_train[r, 0], x_train[r, 1]))

        dist = np.asarray(dist)
        # recovery the index of the n neighbors
        idx = np.argsort(dist)[:seq + 1]
        # If it exists, delete indexes of type m_ii.
        if c in idx:
            idx = idx[idx != c]
        else:
            idx = idx[:seq]

        nearest_dist.append(dist[idx])
        nearest_idx.append(idx)

    return np.asarray(nearest_idx), np.asarray(nearest_dist)

# Recover original data
def recover_original_data(idx_seq, dataset):
    """
    Recovery original data for each point of the sequence.
    :param idx_seq: (n x seq) numpy array with an index of the dataset for each nearest n points.
    :param dataset: (n x features + target) numpy array from original datasets
    :return: (n x seq x features) numpy array.
    """
    recover = dataset[idx_seq]

    return recover

#
def recovery_dist(seq_train, seq_test, x_train, x_test, func):
    """
    Calculates the distance between the features of the nearest n points.
    :param seq: numpy array (n x seq).Sequence of the nearest n points.
    :param X: numpy array (n x features). Features of the dataset
    :return:
    """

    # concatenates the training and test set
    seq = np.concatenate((seq_train, seq_test), axis=0)
    x = np.concatenate((x_train, x_test), axis=0)
    dist = []
    idx = []
    for i in range(seq.shape[0]):
        dist_ind = []
        for row in range(seq.shape[1]):
            d = func(x[i, :], x[seq[i, row], :])
            dist_ind.append(d)
        dist_array = np.asarray(dist_ind)
        sort = np.argsort(dist_array)
        dist.append(dist_array[sort])
        idx.append(seq[i][sort])

    dist = np.asarray(dist)
    idx = np.asarray(idx)

    return idx, dist

def recovery_angle_90(seq_train, seq_test, x_train, x_test, func):
    """
    Calculates the distance between the features of the nearest n points.
    :param seq: numpy array (n x seq).Sequence of the nearest n points.
    :param X: numpy array (n x features). Features of the dataset
    :return:
    """

    # concatenates the training and test set
    seq = np.concatenate((seq_train, seq_test), axis=0)
    x = np.concatenate((x_train, x_test), axis=0)
    dist = []
    idx = []
    for i in range(seq.shape[0]):
        dist_ind = []
        for row in range(seq.shape[1]):
            d = func(x[i, :], x[seq[i, row], :])
            dist_ind.append(d)
        dist_array = np.asarray(dist_ind)
        sort = dist_array.argsort()[::-1]
        dist.append(dist_array[sort])
        idx.append(seq[i][sort])

    dist = np.asarray(dist)
    idx = np.asarray(idx)

    return idx, dist

def recovery_intra_dist(seq_train, x_train, x_test, func = None, geo: bool = True):
    """
    Calculates the distance between the features of the nearest n points.
    :param seq: numpy array (n x seq).Sequence of the nearest n points.
    :param X: numpy array (n x features). Features of the dataset
    :return:
    """

    # concatenates the training and test set
    seq = seq_train
    idx = [i for i in range(seq_train.shape[0])]
    idx = np.asarray(idx).reshape(-1, 1)
    seq = np.hstack((idx, seq))

    x = np.concatenate((x_train, x_test), axis=0)
    dist = []

    for i in range(seq.shape[0]):
        dist_ind = []
        for point in seq[i]:
            for col in seq[i]:
                if geo:
                    d = dist_geodesic(x[point, 0], x[point, 1], x[col, 0], x[col, 1])
                else:
                    d = func(x[point, :], x[col, :])
                dist_ind.append(d)
        dist_array = np.asarray(dist_ind)
        dist.append(dist_array)

    dist = np.asarray(dist).reshape(seq.shape[0], seq.shape[1], seq.shape[1])

    return dist


#########################################################################################
# graph
#########################################################################################

# Create threshold for distance
def dist_threshold(D, t):
    return 0 if D > t else D


# Create distance geodesic matrix
def dist_geodesic_matrix(X):
    X = X[:, 0:2]
    D = np.zeros(X.shape[0], X.shape[0])
    for c in range(X.shape[0]):
        for r in range(X.shape[0]):
            if r <= c:
                D[r, c] = dist_geodesic(X[c, 0], X[c, 1], X[r, 0], X[r, 1])
        print(c)
    D = D.cpu().numpy()
    return D


# Create distance geodesic matrix complement
def dist_geodesic_matrix_complement(X, complement):
    X = X[:, 0:2]
    D = np.zeros(X.shape[0], (X.shape[0] - complement))
    for c in range(X.shape[0] - complement):
        for r in range(X.shape[0]):
            if r <= c + complement:
                D[r, c] = dist_geodesic(X[c + complement, 0], X[c + complement, 1], X[r, 0], X[r, 1])
        print(c)
    D = D.cpu().numpy()
    return D


# Simililarity Measure
def similarity_measure(dist, sigma):
    simi = np.exp(-(dist) / (2 * (sigma ** 2)))
    return simi


# Create similarity matrix

def similarity_matrix(X, sigma, t):
    simi = similarity_measure(X, sigma)
    simi[(simi <= similarity_measure(5, sigma))] = 0
    return simi


def similarity_matrix_old(X, sigma, t):
    M = np.zeros(X.shape[0], X.shape[0])
    for c in range(X.shape[0]):
        for r in range(X.shape[0]):
            if r <= c:
                if X[r, c] >= t:
                    M[r, c] == 0
                else:
                    M[r, c] = similarity_measure(X[r, c], sigma)
    M = M.cpu().numpy()
    return M


# Create edges
def create_edges(G, D, t):
    for r in range(D.shape[0]):
        edges = []
        for c in range(D.shape[0]):
            if r <= c:
                if D[r, c] <= t:
                    string = r, c
                    edges.append(string)
        G.add_edges_from(edges)


# Find row and columns in similarity matrix
def find_similarity_matrix(S1, S2, S3, l1, l2, r, c):
    if r <= l1 and c <= l1:
        s = S1[r, c]
    if r > l1 and r <= l2 and c > l1 and c <= l2:
        s = S2[r, c]
    if r > l2 and c > l2:
        s = S3[r, c]
    return s


# Create wights
def create_wights(G, D, t, S1, S2, S3, l1, l2, ):
    for r in range(D.shape[0]):
        for c in range(D.shape[0]):
            if r <= c:
                if D[r, c] <= t:
                    G[r][c]['weight'] = find_similarity_matrix(S1, S2, S3, l1, l2, r, c)


# Create markov chain
def markov_chain(G, h):
    neighbors = [g for g in G.neighbors(h)]
    total = 0
    weight = []
    for n in range(len(neighbors)):
        weight.append(G[h][neighbors[n]]['weight'])
        total += G[h][neighbors[n]]['weight']
    p = weight / total
    return neighbors, p


# Create random walks to data of training
def random_walks_train(l, m, G_train, node):
    # The set of sequence S
    Seq_train = []

    # Total number of sequence
    L = l

    # Total number of desired sequences
    M = m

    G_train = G_train.subgraph(node)

    # Execute M times this command sequence
    while len(Seq_train) < M:
        Hi = int(np.random.choice(G_train.nodes()))
        while len(G_train[Hi]) == 0:
            Hi = int(np.random.choice(G_train.nodes()))
        # Dictionary that associate nodes with the amount of times it was visited
        Sc = []
        # Randomly pick one node hi and add hi to sc
        Sc.append(Hi)
        Hc = Hi
        # Execute the random walk with size M
        while len(Sc) < L:
            # Visualize the vertex neighborhood
            Hc_Neighbors, p_Neighbors = markov_chain(G_train, Hc)
            # Choose a vertex from the vertex neighborhood to start the next random walk
            Hj = np.random.choice(Hc_Neighbors, p=p_Neighbors)
            while len(G_train[Hj]) == 0:
                Hj = np.random.choice(Hc_Neighbors, replace=True, p=p_Neighbors)
            # add hj to sc
            Sc.append(Hj)
            Hc = Hj
        Seq_train.append(Sc)

    return np.asarray(Seq_train)


# Create random walks to data of test
def random_walks_test(l, m, X_train, X_test, G_test):
    test = [(i + X_train.shape[0]) for i in range(X_test.shape[0])]
    count = 0

    # The set of sequence S
    Seq_test = []

    # Total number of houses
    L = l

    # Total number of desired sequences
    M = m

    # G_train = G_train.subgraph(node)

    # Execute M times this command sequence
    while len(Seq_test) < M:
        Hi = int(np.random.choice(G_test.nodes()))
        while len(G_test[Hi]) == 0:
            Hi = int(np.random.choice(G_test.nodes()))
        # Dictionary that associate nodes with the amount of times it was visited
        Sc = []
        # Randomly pick one node hi and add hi to sc
        Sc.append(Hi)
        Hc = Hi
        # Execute the random walk with size M
        while len(Sc) < L:

            # Visualize the vertex neighborhood
            Hc_Neighbors, p_Neighbors = markov_chain(G_test, Hc)
            # Choose a vertex from the vertex neighborhood to start the next random walk
            Hj = np.random.choice(Hc_Neighbors, p=p_Neighbors)
            while len(G_test[Hj]) == 0:
                Hj = np.random.choice(Hc_Neighbors, replace=True, p=p_Neighbors)
            # add hj to sc
            Sc.append(Hj)
            if len(Sc) == L:
                count = 0
                for i in range(L):
                    if Sc[i] in test:
                        count += 1
            Hc = Hj
            if count == 1:
                Seq_test.append(Sc)
                count = 0

    return np.asarray(Seq_test)


# Create sequence generated by random walks for feed in CNN
def random_walks_cluster(l, m, G_train, ran_inf, ran_sup, node, type):
    # The set of sequence S
    Seq_train = []

    # Total number of sequence
    L = l

    # Total number of desired sequences
    M = m

    G = G_train.subgraph(node)

    # Execute M times this command sequence
    while len(Seq_train) < M:
        for i in range(ran_inf, ran_sup):
            Hi = i
            # Dictionary that associate nodes with the amount of times it was visited
            Sc = []
            # Randomly pick one node hi and add hi to sc
            Sc.append(Hi)
            Hc = Hi
            # Execute the random walk with size M
            while len(Sc) < L:
                # Visualize the vertex neighborhood
                Hc_Neighbors, p_Neighbors = markov_chain(G, Hc)
                # Choose a vertex from the vertex neighborhood to start the next random walk
                Hj = np.random.choice(Hc_Neighbors, p=p_Neighbors)
                if type == 'test':
                    while Hj in range(ran_inf, ran_sup):
                        Hj = np.random.choice(Hc_Neighbors, p=p_Neighbors)
                # while len(G_train[Hj]) == 0:
                # Hj = np.random.choice(Hc_Neighbors, replace=True, p=p_Neighbors)
                # add hj to sc
                Sc.append(Hj)
                # delete for not change the cluster of neighborhood Hc = Hj
            Seq_train.append(Sc)

    return np.asarray(Seq_train)


def recovery_dist_eucledan_segm(seq, X):
    dist_cnn_ind = []
    dist_cnn = []
    for i in range(seq.shape[0]):
        dist_cnn_ind = []
        for l in range(seq.shape[1] - 1):
            d = distance.euclidean(X[seq[i, l], :], X[seq[i, l + 1], :])
            dist_cnn_ind.append(d)
        dist_cnn.append(dist_cnn_ind)
    return np.asarray(dist_cnn)


def recovery_media_preco_segm(seq, X):
    dist_cnn_ind = []
    dist_cnn = []
    for i in range(seq.shape[0]):
        dist_cnn_ind = []
        for l in range(seq.shape[1] - 1):
            d = np.mean([X[seq[i, l]], X[seq[i, l + 1]]])
            dist_cnn_ind.append(d)
        dist_cnn.append(dist_cnn_ind)
    return np.asarray(dist_cnn)


def recovery_dist_geo_segm(X, X_train_test):
    dist_cnn_ind = []
    dist_cnn = []
    for i in range(X.shape[0]):
        dist_cnn_ind = []
        for l in range(X.shape[1] - 1):
            dist_cnn_ind.append(
                dist_geodesic(X_train_test[X[i, l], 0], X_train_test[X[i, l], 1], X_train_test[X[i, l + 1], 0], \
                              X_train_test[X[i, l + 1], 1]))
        dist_cnn.append(dist_cnn_ind)
    return np.asarray(dist_cnn)


# Recover original data for random walker
def recovery_dist_geo(X, X_train_test, r):
    dist_cnn_ind = []
    dist_cnn = []
    for i in range(X.shape[0]):
        dist_cnn_ind = []
        for l in range(X.shape[1]):
            dist_cnn_ind.append(dist_geodesic(X_train_test[i + r, 0], X_train_test[i + r, 1], X_train_test[X[i, l], 0], \
                                              X_train_test[X[i, l], 1]))
        dist_cnn.append(dist_cnn_ind)
    return np.asarray(dist_cnn)

# Order sequence according to distance
"""
def order_sequence(simi, seq_geo_dist, seq_geo):
    simi = train_x_dist_cosine_0
    nearest = []
    nearest_dist = []
    for i in range(seq_geo_dist.shape[0]):
        n = np.argsort(simi[i])
        nearest.append(seq_geo[i][n])
        nearest_dist.append(simi[i][n])
    nearest_house_train_cosine = np.asarray(nearest)
    nearest_dist_train_cosine = np.asarray(nearest_dist)
    return arest_house_train_cosine, nearest_dist_train_cosine
"""


# Preparing the data for evaluate and Obtaining the positioning of each test data

def get_positioning_data(X_test, Seq_test, X_train):
    list_test = []
    l = []
    for i in range(X_test.shape[0]):
        for s in range(Seq_test.shape[0]):
            list_test = []
            if i + X_train.shape[0] in Seq_test[s]:
                list_test.append(i + X_train.shape[0])
                list_test.append(s)
                list_test.append(list(Seq_test[s]).index(i + X_train.shape[0]))
                l.append(list_test)
    return l


# Create a tensor with shape (n, length, width, channel)
def grid_position(X, seq, channel):
    """
    param:
    X = base of data
    seq = number of object in the sequence
    channel = number of feature in the database
    """
    grid = []
    grid_all = []
    for j in range(X.shape[0]):
        for i in range(len(X[0])):
            long_posi = np.where(np.unique(X[0, :, 1]) == X[0][i][1])[0][0]
            lat_posi = np.where(np.unique(X[0, :, 0]) == X[0][i][0])[0][0]
            grid.append(long_posi)
            grid.append(lat_posi)
        grid_all.append(grid)
        grid = []

    posi = np.reshape(np.array(grid_all), (X.shape[0], seq, 2))
    zeros_grid = []
    for e in range(channel):
        for j in range(X.shape[0]):
            zeros = np.zeros((seq, seq))
            for i in range(posi[j].shape[0]):
                idx_x = posi[j, i, 0]  # 0 is longitude
                idx_y = posi[j, i, 1]  # 1 is latitude
                zeros[idx_x, idx_y] = X[j, i, e]  # 1 is longitude
            zeros_grid.append(zeros)

    return np.reshape(np.array(zeros_grid), (X.shape[0], seq, seq, channel))
