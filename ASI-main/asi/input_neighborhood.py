from tensorflow.keras.layers import Input


def getcontext(shape_context_struc_target, shape_context_struc_w_lat_long,
               shape_context_geo_target_dist, num_nearest,
               geo, euclidean, num_nearest_geo, num_nearest_eucli):
    """

    :param shape_context_struc_target:
    :param shape_context_struc_w_lat_long:
    :param shape_context_geo_target_dist:
    :param num_nearest:
    :param geo:
    :param euclidean:
    :param num_nearest_geo:
    :param num_nearest_eucli:
    :return:
    """

    ######################## Input of context ########################

    if geo:

        # original data from the nearest points plus target and distance (m, seq , features + target + dist)
        context_geo_target_dist = Input(shape=(num_nearest_geo, shape_context_geo_target_dist),
                                        name='context_geo_target_dist')
        # geodesic distance from nearest points (m, seq)
        dist_geo = Input(shape=(num_nearest_geo), name='dist_geo')

    else:
        context_geo_target_dist = 0
        dist_geo = 0

    # original data from the nearest points less lat and long (m, seq , features - 2)
    context_struc_without_lat_long = Input(shape=(num_nearest, shape_context_struc_w_lat_long),
                                           name='context_phe_w_lat_long')

    if euclidean:
        # original data from the nearest points plus target (m, seq , features + target)
        context_struc_eucli_target = Input(shape=(num_nearest_eucli, shape_context_struc_target),
                                           name='context_struc_eucl_target')
        # euclidean distance from nearest points (m, seq)
        dist_eucli = Input(shape=(num_nearest_eucli,), name='dist_eucli')
    else:
        context_struc_eucli_target = 0
        dist_eucli = 0

    ####################################################################

    return context_geo_target_dist, context_struc_without_lat_long, context_struc_eucli_target, \
           dist_geo, dist_eucli
