
import tensorflow as tf
from tensorflow.keras.layers import Lambda, subtract, multiply

from utils.utils import srs


class Distance:
    """

    """

    def __init__(self, phenomenon_structure_repeat, context_structure, type_distance):
        """

        :param phenomenon_structure_repeat: repeats the phenomenon data according to the number of closest neighbors
        :param context_structure: original data of the nearest phenomenon less lat e long (m, seq , features - 2)
        :param type_distance: the type of distance that was used
        """

        self.structural_dada = phenomenon_structure_repeat
        self.structural_context = context_structure
        self.type_distance = type_distance
        self.choices = {
            'euclidean': self.euclidean
        }

    def run(self):

        choice = self.type_distance
        action = self.choices.get(choice)
        if action:

            distance = action()

        else:
            distance = "{0} is not a valid choice $$$$$$$$$$$$$$$".format(choice)
            print(distance)

        return distance

    def euclidean(self):

        # calculates the subtraction of each point about each nearest neighbor (m, seq , features - 2)
        euclidean = subtract([self.structural_dada, self.structural_context])

        # calculates the Euclidean distance (m, seq, 1)
        euclidean = srs(euclidean)

        # adjust the dimensions (m, seq)
        euclidean = Lambda(lambda x: tf.math.reduce_sum(x, axis=2))(euclidean)

        return euclidean
