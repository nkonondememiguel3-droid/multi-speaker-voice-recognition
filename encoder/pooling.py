import tensorflow as tf


class AttentiveStatisticsPooling(tf.keras.layers.Layer):
    """
    Attentive Statistics Pooling (ASP).

    Instead of averaging all time frames equally, ASP learns a per-frame
    attention weight α_t, then computes a weighted mean and weighted std.
    The two are concatenated, so the pooling output is always 2× the
    input channel dimension.

    For an input of shape (batch, T, C):
        α_t  = softmax(tanh(W * h_t + b))       # (batch, T, 1)
        μ    = Σ α_t * h_t                       # (batch, C)
        σ    = sqrt(Σ α_t * h_t² − μ²)          # (batch, C)
        out  = concat([μ, σ])                    # (batch, 2C)

    The weighted std captures speaking-rate and energy variation across
    the utterance — information that a plain mean would discard.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, T, C)
        C = input_shape[-1]
        # Single linear attention head: maps C-dim frame → scalar score
        self.attention = tf.keras.layers.Dense(1, use_bias=True)
        super().build(input_shape)

    def call(self, x, training=False):
        # x: (batch, T, C)

        # Attention scores → softmax over time axis
        e = tf.nn.tanh(self.attention(x))        # (batch, T, 1)
        alpha = tf.nn.softmax(e, axis=1)         # (batch, T, 1)

        # Weighted mean
        mu = tf.reduce_sum(alpha * x, axis=1)    # (batch, C)

        # Weighted variance → std
        # E[x²] - E[x]²  (computed in weighted form)
        mu_sq = mu ** 2                                          # (batch, C)
        ex2 = tf.reduce_sum(alpha * (x ** 2), axis=1)         # (batch, C)
        var = tf.nn.relu(ex2 - mu_sq)                         # clamp ≥ 0
        sigma = tf.sqrt(var + 1e-9)                             # (batch, C)

        # Concatenate mean and std
        return tf.concat([mu, sigma], axis=-1)   # (batch, 2C)

    def get_config(self):
        return super().get_config()
