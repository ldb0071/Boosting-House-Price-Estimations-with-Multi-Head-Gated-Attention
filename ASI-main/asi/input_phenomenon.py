
from tensorflow.keras.layers import Input, RepeatVector


def getinput(shape_input_phe, shape_input_phe_w_lat_long, num_nearest):

    """

    :param shape_input_phe:
    :param shape_input_phe_w_lat_long:
    :param num_nearest:
    :return:
    """

    ######################## Input of phenomeno ########################

    # the phenomena input features X_train.shape[1] (m, features)
    input_phenomenon = Input(shape=(shape_input_phe,), name='input_phenomenon')

    # the phenomena input features without lat and long (m, features - 2)
    input_phenomenon_without_lat_long = Input(shape=(shape_input_phe_w_lat_long,), name='input_phe_w_lat_long')

    # repeats the phenomenon data according to the number of closest neighbors
    repeat = RepeatVector(num_nearest, name='repeat_phe_w_lat_long')(input_phenomenon_without_lat_long)

     ######################################################################

    return input_phenomenon, input_phenomenon_without_lat_long, repeat
