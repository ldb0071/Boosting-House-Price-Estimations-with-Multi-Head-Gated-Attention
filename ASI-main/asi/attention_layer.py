from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

import tensorflow as tf
from tensorflow.keras.layers import multiply, RepeatVector
from tensorflow.keras.layers import Lambda, Permute
from asi.distance import Distance
from asi.transformation import CompFunction


class Attention(Layer):

    def __init__(self,
                 sigma,
                 num_nearest,
                 shape_input_phenomenon,
                 type_compatibility_function,
                 num_features_extras,
                 calculate_distance=False,
                 graph_label=None,
                 phenomenon_structure_repeat=None,
                 context_structure=None,
                 type_distance=None,
                 suffix_mean=None,
                 **kwargs):

        self.sigma = sigma,
        self.num_nearest = num_nearest,
        self.shape_input_phenomenon = shape_input_phenomenon,
        self.type_compatibility_function = type_compatibility_function,
        self.num_features_extras = num_features_extras,
        self.calculate_distance = calculate_distance,
        self.graph_label = graph_label,
        self.phenomenon_structure_repeat = phenomenon_structure_repeat,
        self.context_structure = context_structure,
        self.type_distance = type_distance,
        self.suffix_mean = suffix_mean
       

        self.output_dim = self.shape_input_phenomenon[0] + self.num_features_extras[0]

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        # Initialize weights for each attention head
        # Layer kernel
        self.kernel = self.add_weight(shape=(self.num_nearest[0], self.num_nearest[0]),
                                 name='kernel_{}'.format(self.graph_label[0]))

        #Layer bias
        self.bias = self.add_weight(shape=(self.num_nearest[0],),
                               name='bias_{}'.format(self.graph_label[0]))

        self.built = True

    def call(self, inputs):
        source_distance = inputs[0]  # Node features (N x F)
        context = inputs[1]

        ######################## Attention data ########################

        if self.calculate_distance[0]:

            dist = Distance(self.phenomenon_structure_repeat[0], self.context_structure[0], self.type_distance[0])
            distance = dist.run()

        else:
            distance = source_distance

        # calculate the similarity measure of each neighbor (m, seq)
        comp_func = CompFunction(self.sigma[0], distance, self.type_compatibility_function[0], self.graph_label[0])
        simi = comp_func.run()

        # calculates the weights associated with each neighbor (m, seq)
        weight = K.dot(simi, self.kernel)
        weight = K.bias_add(weight, self.bias)
        weight = K.softmax(weight)

        # repeats the previous vector as many times as the feature number plus the point target and features extras
        # (input_phenomenon + 1 + num_features_extras, seq)
        prob_repeat = RepeatVector(self.shape_input_phenomenon[0] + self.num_features_extras[0])(weight)

        # inverts the dimensions in such a way that in each line,
        # we have the weight assigned to the neighbor (seq, input_phenomenon + 1 + num_features_extras)
        prob_repeat = Permute((2, 1))(prob_repeat)

        # multiplies each neighbor's feature by its respective weight
        # (seq, input_phenomenon + 1 + num_features_extras) x (seq, input_phenomenon + 1 + num_features_extras)
        relevance = multiply([prob_repeat, context])

        # add each column to find the mean vector (input_phenomenon + 1 + num_features_extras,)
        mean = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), name=self.suffix_mean[0])(relevance)

        return mean

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
