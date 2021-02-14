import tensorflow as tf

def dist_loss(data, min_dist, max_dist = 20):
    pairwise_dist = cdisttf(data, data)
    dist = pairwise_dist - min_dist
    bigdist = max_dist - pairwise_dist
    loss = tf.math.exp(-dist) + tf.math.exp(-bigdist)
    return loss

def cdisttf(data_1, data_2):
    prod = tf.math.reduce_sum(
            (tf.expand_dims(data_1, 1) - tf.expand_dims(data_2, 0)) ** 2, 2
        )
    return (prod + 1e-10) ** (1 / 2)



