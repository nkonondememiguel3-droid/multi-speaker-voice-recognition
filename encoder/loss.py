import tensorflow as tf


class SubcenterArcFace(tf.keras.layers.Layer):
    """
    SubcenterArcFace loss layer for speaker verification.

    Inherits from Layer (not Loss) so that the weight matrix W is
    properly registered in the model's trainable_variables and updated
    by the optimizer during training.

    The layer takes embeddings as input and returns a scalar loss.
    It is called inside MultiTaskECAPA.compute_loss(), not as a
    standalone Keras loss function.

    K sub-centers per class allow the model to handle intra-speaker
    variability (recording conditions, emotion, speaking style) without
    collapsing all variation into a single class center.

    Forward pass:
        1. L2-normalize W column-wise  (each sub-center on unit sphere)
        2. cosine = embeddings @ W_norm         (batch, n_spk * K)
        3. reshape + max over K sub-centers  →  (batch, n_spk)
        4. add angular margin m to target class angle
        5. scale by s → sparse cross-entropy
    """

    def __init__(self, n_speakers: int, embedding_dim: int = 192,
                 K: int = 3, margin: float = 0.2, scale: float = 32.0,
                 **kwargs):
        kwargs['dtype'] = 'float32'   # ← add this
        super().__init__(**kwargs)
        self.n_speakers = n_speakers
        self.embedding_dim = embedding_dim
        self.K = K
        self.margin = margin
        self.scale = scale

    def build(self, input_shape):
        """
        Weights are created in build() rather than __init__() — this is
        the correct Keras pattern. build() is called automatically on the
        first forward pass once the input shape is known.
        """
        self.W = self.add_weight(
            name='subcenter_weights',
            shape=(self.embedding_dim, self.n_speakers * self.K),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, speaker_ids, embeddings, training=False):
        # Cast embeddings to float32 — ArcFace margin arithmetic (arccos, cos)
        # must run in full precision regardless of mixed precision policy.
        # float16 has insufficient range for angles near 0 and π.
        embeddings = tf.cast(embeddings, tf.float32)

        # Normalize sub-center vectors onto unit sphere
        W_norm = tf.math.l2_normalize(self.W, axis=0)        # (192, n_spk * K)

        # Cosine similarities
        # (batch, n_spk * K)
        cosine = tf.matmul(embeddings, W_norm)

        # Sub-center max: assign each sample to its nearest sub-center
        cosine = tf.reshape(cosine, (-1, self.n_speakers, self.K))
        cosine = tf.reduce_max(cosine, axis=-1)              # (batch, n_spk)

        # Clip for numerical safety before arccos
        cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # One-hot encode target speaker — explicitly float32
        speaker_ids = tf.cast(speaker_ids, tf.int32)
        one_hot = tf.cast(
            tf.one_hot(speaker_ids, self.n_speakers), tf.float32
        )

        # Add angular margin to target class only
        theta = tf.acos(cosine)
        target_cos = tf.cos(theta + self.margin)

        # Substitute margin-penalized cosine for target class
        logits = cosine * (1.0 - one_hot) + target_cos * one_hot
        logits = logits * self.scale

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=speaker_ids, logits=logits
        )
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'n_speakers':    self.n_speakers,
            'embedding_dim': self.embedding_dim,
            'K':             self.K,
            'margin':        self.margin,
            'scale':         self.scale,
        })
        return cfg
