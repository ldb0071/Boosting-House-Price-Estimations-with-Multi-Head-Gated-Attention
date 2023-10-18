
import tensorflow as tf
from tensorflow.keras.layers import Lambda


class CompFunction:
    """

    """

    def __init__(self, sigma, matrix, type_compat_function, graph_label):

        """

        :param sigma:
        :param matrix:
        :param type_compat_function:
        :param graph_label:
        """

        self.sigma = sigma
        self.matrix = matrix
        self.type_compat_function = type_compat_function
        self.graph_label = graph_label
        self.choices = {
            'kernel_gaussiano': self.kernelgaussiano,
            'identity': self.identity
        }

    def run(self):

        choice = self.type_compat_function
        action = self.choices.get(choice)
        if action:

            similarity = action()

        else:
            similarity = "{0} is not a valid choice".format(choice)
            print(similarity)

        return similarity

    def kernelgaussiano(self):

        # calcula a medida de similarida de cada vizinho (m,s1)
        similarity = Lambda(lambda x: tf.math.exp(-x * (self.sigma ** 2/2)),
                            name=self.graph_label + '_gaussiano')(self.matrix)

        return similarity

    def identity(self):

        # calcula a medida de similarida de cada vizinho (m,s1)
        similarity = Lambda(lambda x: x, name=self.graph_label + '_identity')(self.matrix)

        return similarity

