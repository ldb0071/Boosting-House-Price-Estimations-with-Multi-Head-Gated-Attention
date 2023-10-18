import numpy as np
import utils.utilsgeo as ug
from sklearn.preprocessing import StandardScaler
from config import PATH
import copy


class Geds:
    """

    """

    def __init__(self, id_dataset: str, num_nearest: int, geo: bool = True, euclidean: bool = True,
                 sequence: str = '', scale: bool = True, input_target_context=True,
                 input_dist_context_geo=True, input_dist_context_eucl=False, scale_euclidean=True,
                 scale_geo=False):
        """

        :param id_dataset:
        :param num_nearest:
        :param geo:
        :param euclidean:
        :param sequence:
        :param scale:
        :param input_target_context:
        :param input_dist_context_geo:
        :param input_dist_context_eucl:
        :param scale_euclidean:
        :param scale_geo:
        """


        self.id_dataset = id_dataset
        self.scale = scale
        self.num_nearest = num_nearest
        self.geo = geo
        self.euclidean = euclidean
        self.sequence = sequence
        self.input_target_context = input_target_context
        self.input_dist_context_geo = input_dist_context_geo
        self.input_dist_context_eucl = input_dist_context_eucl
        self.scale_euclidean = scale_euclidean
        self.scale_geo = scale_geo

    def __call__(self):

        assert isinstance(self.id_dataset, object)

        data = np.load(PATH + '/datasets/'+ self.id_dataset + '/data'+self.sequence+'.npz', allow_pickle=True)

        # original data
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        if self.geo:

            # the sequence of the nearest points (geodesic distance)
            nearest_train = data['idx_geo'][:X_train.shape[0], :self.num_nearest]
            nearest_dist_train = data['dist_geo'][:X_train.shape[0], :self.num_nearest]
            nearest_test = data['idx_geo'][X_train.shape[0]:, :self.num_nearest]
            nearest_dist_test = data['dist_geo'][X_train.shape[0]:, :self.num_nearest]

        else:
            nearest_train = 0
            nearest_dist_train = 0
            nearest_test = 0
            nearest_dist_test = 0

        if self.euclidean:

            # the sequence of the nearest points (Euclidean distance employing
            # the geographic distance for the closest points )
            nearest_train_eucli = data['idx_eucli'][:X_train.shape[0], :self.num_nearest]
            nearest_dist_train_eucli = data['dist_eucli'][:X_train.shape[0], :self.num_nearest]
            nearest_test_eucli = data['idx_eucli'][X_train.shape[0]:, :self.num_nearest]
            nearest_dist_test_eucli = data['dist_eucli'][X_train.shape[0]:, :self.num_nearest]
        else:
            nearest_train_eucli = 0
            nearest_dist_train_eucli = 0
            nearest_test_eucli = 0
            nearest_dist_test_eucli = 0

        # Concatenate the data
        X_train_test = np.concatenate((X_train, X_test), axis=0)
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        y_train_scale = copy.deepcopy(y_train)
        y_test_scale = copy.deepcopy(y_test)

        # preprocessing dataset

        scale = self.scale

        dist_train = copy.deepcopy(nearest_dist_train)
        dist_test = copy.deepcopy(nearest_dist_test)
        dist_train_eucli = copy.deepcopy(nearest_dist_train_eucli)
        dist_test_eucli = copy.deepcopy(nearest_dist_test_eucli)

        # Scaler
        if scale:

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            if self.geo:
                nearest_dist_train = scaler.fit_transform(nearest_dist_train)
                nearest_dist_test = scaler.fit_transform(nearest_dist_test)
                if self.scale_geo:
                    dist_train = scaler.fit_transform(dist_train)
                    dist_test = scaler.fit_transform(dist_test)

            if self.euclidean:
                nearest_dist_train_eucli = scaler.fit_transform(nearest_dist_train_eucli)
                nearest_dist_test_eucli = scaler.fit_transform(nearest_dist_test_eucli)

                if self.scale_euclidean:
                    dist_train_eucli = scaler.fit_transform(dist_train_eucli)
                    dist_test_eucli = scaler.fit_transform(dist_test_eucli)


            # Scaled because the target is used in neural networks
            y_train_scale = scaler.fit_transform(y_train_scale.reshape(-1, 1))

        else:

            y_train_scale = y_train_scale.reshape(-1, 1)

        # Training and test
        if self.geo:
            # Recover original data and include the target in sequence
            if self.input_target_context:
                train_x = ug.recover_original_data(nearest_train, np.concatenate((X_train, y_train_scale), axis=1))
                test_x = ug.recover_original_data(nearest_test, np.concatenate((X_train, y_train_scale), axis=1))

            else:
                train_x = ug.recover_original_data(nearest_train, X_train)
                test_x = ug.recover_original_data(nearest_test, X_train)

            # Concatenate the distance to the sequence
            if self.input_dist_context_geo:
            # Geo
                train_x = np.concatenate((train_x, np.reshape(nearest_dist_train, (
                    nearest_dist_train.shape[0], nearest_dist_train.shape[1], 1))), axis=2)
                test_x = np.concatenate((test_x, np.reshape(nearest_dist_test, (
                    nearest_dist_test.shape[0], nearest_dist_test.shape[1], 1))), axis=2)


        else:
            train_x = 0
            test_x = 0

        if self.euclidean:
            # Recover original data and include the target in sequence
            if self.input_target_context:
                train_x_eucli = ug.recover_original_data(nearest_train_eucli,
                                                         np.concatenate((X_train, y_train_scale), axis=1))
                test_x_eucli = ug.recover_original_data(nearest_test_eucli,
                                                        np.concatenate((X_train, y_train_scale), axis=1))
            else:
                train_x_eucli = ug.recover_original_data(nearest_train_eucli, X_train)
                test_x_eucli = ug.recover_original_data(nearest_test_eucli, X_train)
            # Concatenate the distance to the sequence
            if self.input_dist_context_eucl:
                # Euclidean
                train_x_eucli = np.concatenate((train_x_eucli, np.reshape(nearest_dist_train_eucli, (
                    nearest_dist_train_eucli.shape[0], nearest_dist_train_eucli.shape[1], 1))), axis=2)
                test_x_eucli = np.concatenate((test_x_eucli, np.reshape(nearest_dist_test_eucli, (
                    nearest_dist_test_eucli.shape[0], nearest_dist_test_eucli.shape[1], 1))), axis=2)
        else:
            train_x_eucli = 0
            test_x_eucli = 0

        # A sequence with original data of the nearest points more target (m, seq , features + target)
        if self.euclidean:
            context_struc_eucli_target_train = train_x_eucli[:, :, :]  # train_x_p
            context_struc_eucli_target_test = test_x_eucli[:, :, :]  # test_x_p
            # Distance to the nearest targets (euclidean) (m, seq)
            dist_eucli_train = dist_train_eucli  # train_x_e
            dist_eucli_test = dist_test_eucli  # test_x_e
        else:
            context_struc_eucli_target_train = 0
            context_struc_eucli_target_test = 0
            dist_eucli_train = 0
            dist_eucli_test = 0

        # Sequence with original data of the nearest phenomena more target and distance (m, seq , features + target +
        # dist)
        if self.geo:
            context_geo_target_dist_train = train_x[:, :, :]  # train_x_d
            context_geo_target_dist_test = test_x[:, :, :]  # test_x_d
            # Distance to the nearest targets (geodesica) (m, seq)
            dist_geo_train = dist_train  # train_x_g
            dist_geo_test = dist_test  # test_x_g


        else:
            context_geo_target_dist_train = 0
            context_geo_target_dist_test = 0
            dist_geo_train = 0
            dist_geo_test = 0

        # Original data are: X_train, X_test

        return context_struc_eucli_target_train, context_struc_eucli_target_test, \
               context_geo_target_dist_train, context_geo_target_dist_test, dist_geo_train, dist_geo_test,\
               dist_eucli_train, dist_eucli_test, X_train, X_test, y_train, y_test, y_train_scale