
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import multiply, RepeatVector,Multiply,Dense,LayerNormalization
from tensorflow.keras.layers import Lambda, Permute
from tensorflow.keras.initializers import glorot_uniform
from asi.distance import Distance
from asi.transformation import CompFunction
import os

os.environ['PYTHONHASHSEED'] = '0'

tf.random.set_seed(42)


class Attention_2(Layer):

    def __init__(self, 
                 sigma,
                 num_nearest,
                 shape_input_phenomenon,
                 type_compatibility_function,
                 num_features_extras,
                 calculate_distance=False,
                 graph_label=None,
                 Num_heads=int,
                 phenomenon_structure_repeat=None,
                 context_structure=None,
                 type_distance=None,
                 suffix_mean=None,
                 **kwargs):

        super(Attention_2, self).__init__(**kwargs)
       
        self.sigma = sigma
        self.num_heads = Num_heads
        self.num_nearest = num_nearest
        self.shape_input_phenomenon = shape_input_phenomenon
        self.type_compatibility_function = type_compatibility_function
        self.num_features_extras = num_features_extras
        self.calculate_distance = calculate_distance
        self.graph_label = graph_label
        self.phenomenon_structure_repeat = phenomenon_structure_repeat
        self.context_structure = context_structure
        self.type_distance = type_distance
        self.suffix_mean = suffix_mean
  
        self.seed = 23
        self.initializer = glorot_uniform(seed=self.seed)


        if isinstance(self.shape_input_phenomenon, int):
            shape_val = self.shape_input_phenomenon
        else:
            shape_val = self.shape_input_phenomenon[0]

        if isinstance(self.num_features_extras, int):
            features_val = self.num_features_extras
        else:
            features_val = self.num_features_extras[0]

        self.output_dim = shape_val + features_val

    def build(self, input_shape):
        self.all_kernels = []
        self.all_biases = []

        if isinstance(self.num_nearest, int):
            num_nearest_val = self.num_nearest
        else:
            num_nearest_val = self.num_nearest[0]

        if isinstance(self.graph_label, str):
            graph_label_val = self.graph_label
        else:
            graph_label_val = self.graph_label[0]
        
        for _ in range(self.num_heads):
            kernel = self.add_weight(shape=(num_nearest_val, num_nearest_val),
                                     initializer=self.initializer,
                                     name='kernel_head_{}'.format(_))
            self.all_kernels.append(kernel)

            bias = self.add_weight(shape=(num_nearest_val,),
                                   initializer=self.initializer,
                                   name='bias_head_{}'.format(_))
            self.all_biases.append(bias)

        self.gate_weights = self.add_weight(shape=(self.num_heads,),
                                            initializer=self.initializer,
                                            name='gate_weights')
        self.gate_bias = self.add_weight(shape=(self.num_heads,),
                                         initializer=self.initializer,
                                         name='gate_bias')
        self.gate = Dense(num_nearest_val, activation='sigmoid', name='gate_{}'.format(graph_label_val))


        self.built = True

    def call(self, inputs):
        all_outputs = []

        source_distance = inputs[0]
        context = inputs[1]

        if isinstance(self.shape_input_phenomenon, int):
            shape_val = self.shape_input_phenomenon
        else:
            shape_val = self.shape_input_phenomenon[0]

        if isinstance(self.num_features_extras, int):
            features_val = self.num_features_extras
        else:
            features_val = self.shape_input_phenomenon[0] + self.num_features_extras[0]

        for i in range(self.num_heads):
            kernel = self.all_kernels[i]
            bias = self.all_biases[i]

            if self.calculate_distance:
                dist = Distance(self.phenomenon_structure_repeat, self.context_structure, self.type_distance)
                distance = dist.run()
            else:
                distance = source_distance


            comp_func = CompFunction(self.sigma, distance, self.type_compatibility_function, self.graph_label)
    
            simi = comp_func.run()
            
            weight = K.dot(simi, kernel)
            weight = K.bias_add(weight, bias)
            weight = K.softmax(weight)
            
            gate_weights = self.gate(weight)
            gated_weight = Multiply()([weight, gate_weights])
      
            prob_repeat = RepeatVector(shape_val + features_val)(gated_weight)
            prob_repeat = Permute((2, 1))(prob_repeat)
            relevance = multiply([prob_repeat, context])
            mean = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), name=self.suffix_mean)(relevance)

            all_outputs.append(mean)
        if self.num_heads > 1:
            gate = tf.nn.softmax(self.gate_weights + self.gate_bias)
            output = tf.add_n([gate[i] * all_outputs[i] for i in range(self.num_heads)])
        else:
            output =mean

        


        # Normalizing the summed output


        return  output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)