import tensorflow as tf
from encoder.pooling import AttentiveStatisticsPooling


# ─────────────────────────────────────────────────────────────────────────────
# SE (Squeeze-and-Excitation) block
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation channel attention.

    Globally averages across time to produce a channel descriptor,
    passes it through two FC layers (bottleneck then restore),
    then multiplies back onto the input frame-wise.

    This lets the network reweight which frequency channels matter
    for a given utterance — e.g. suppress low-frequency noise channels
    when the speaker has a high-pitched voice.

    reduction controls the bottleneck size: C → C//reduction → C.
    """

    def __init__(self, channels: int, reduction: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(
            channels // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, x, training=False):
        # x: (batch, T, C)
        # Squeeze: global average across time
        s = tf.reduce_mean(x, axis=1)            # (batch, C)
        # Excitation: two-layer bottleneck
        s = self.fc1(s)                          # (batch, C//reduction)
        s = self.fc2(s)                          # (batch, C)
        # Scale: broadcast back over time
        s = tf.expand_dims(s, axis=1)            # (batch, 1, C)
        return x * s                             # (batch, T, C)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'channels': self.fc2.units,
                    'reduction': self.fc1.units})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Res2Net multi-scale grouped convolution
# ─────────────────────────────────────────────────────────────────────────────

class Res2NetBlock(tf.keras.layers.Layer):
    """
    Res2Net hierarchical residual connections.

    Splits the C channels into `scale` equal groups. Each group's output
    is added to the next group's input before convolution, creating a
    multi-scale receptive field within a single layer without increasing
    parameters significantly.

    For scale=8 and C=512: each branch processes C//scale = 64 channels.

    Branch computation:
        y_1 = conv(x_1)
        y_2 = conv(x_2 + y_1)
        y_3 = conv(x_3 + y_2)
        ...
        y_s = conv(x_s + y_{s-1})

    The first branch has no addition (standard residual start).
    """

    def __init__(self, channels: int, scale: int = 8,
                 kernel_size: int = 3, dilation: int = 1, **kwargs):
        super().__init__(**kwargs)
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale = scale
        self.branch_dim = channels // scale

        # One conv per branch (skip first branch — it's identity)
        self.convs = [
            tf.keras.layers.Conv1D(
                self.branch_dim,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                padding='same',
                activation=None,
                use_bias=False,
            )
            for _ in range(scale - 1)
        ]
        self.bns = [
            tf.keras.layers.BatchNormalization() for _ in range(scale - 1)
        ]

    def call(self, x, training=False):
        # x: (batch, T, C)
        # Split into `scale` chunks along channel axis
        # list of (batch, T, C//scale)
        chunks = tf.split(x, self.scale, axis=-1)

        outputs = []
        prev = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First branch: pass through unchanged
                outputs.append(chunk)
                prev = chunk
            else:
                # Add previous branch output before conv
                h = chunk + prev
                h = self.convs[i - 1](h, training=training)
                h = self.bns[i - 1](h, training=training)
                h = tf.nn.relu(h)
                outputs.append(h)
                prev = h

        return tf.concat(outputs, axis=-1)    # (batch, T, C)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'scale': self.scale,
                    'branch_dim': self.branch_dim})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# SE-Res2Block — one full ECAPA block
# ─────────────────────────────────────────────────────────────────────────────

class SERes2Block(tf.keras.layers.Layer):
    """
    One ECAPA-TDNN block:

        input
          │
          ├─ 1×1 pointwise conv (expand to channels)
          │
          ├─ Res2Net multi-scale dilated conv
          │
          ├─ 1×1 pointwise conv (project back)
          │
          ├─ BatchNorm + ReLU
          │
          ├─ SE channel attention
          │
          └─ residual add
          │
        output

    The residual connection requires input and output to have the same
    channel dimension, which the 1×1 convolutions enforce.

    dilation controls the receptive field:
        Block 2: dilation=2  → 5-frame effective context
        Block 3: dilation=3  → 7-frame effective context
        Block 4: dilation=4  → 9-frame effective context
        Block 5: dilation=5  → 11-frame effective context
    """

    def __init__(self, channels: int, scale: int = 8,
                 kernel_size: int = 3, dilation: int = 1,
                 reduction: int = 8, **kwargs):
        super().__init__(**kwargs)

        self.pw1 = tf.keras.layers.Conv1D(
            channels, 1, padding='same', use_bias=False)
        self.res2 = Res2NetBlock(channels, scale=scale,
                                 kernel_size=kernel_size, dilation=dilation)
        self.pw2 = tf.keras.layers.Conv1D(
            channels, 1, padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(channels, reduction=reduction)

    def call(self, x, training=False):
        residual = x

        h = self.pw1(x,    training=training)
        h = tf.nn.relu(h)
        h = self.res2(h,   training=training)
        h = self.pw2(h,    training=training)
        h = self.bn(h,     training=training)
        h = tf.nn.relu(h)
        h = self.se(h,     training=training)

        return h + residual     # (batch, T, C)

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Full ECAPA-TDNN model
# ─────────────────────────────────────────────────────────────────────────────

class ECAPA_TDNN(tf.keras.Model):
    """
    ECAPA-TDNN speaker encoder.

    Input:  (batch, 80, T)  — log-Mel spectrogram, channels-first
    Output: (batch, 192)    — L2-normalized speaker embedding

    Architecture:
        Block 1:  Conv1D stem, 512 channels, kernel=5
        Blocks 2-5: SE-Res2Block with increasing dilation (2,3,4,5)
        MFA:      Multi-scale Feature Aggregation — concat all block outputs
                  then project 4×512=2048 → 1536 with a 1×1 conv
        ASP:      Attentive Statistics Pooling → 3072-dim vector
        FC:       Dense(192) + BatchNorm → L2-normalized embedding

    The MFA step is what makes ECAPA distinctive: rather than using only
    the last block's output, all intermediate representations are combined.
    Each block captures different temporal scales (due to different dilations),
    so their concatenation gives a richer feature for pooling.
    """

    def __init__(self, embedding_dim: int = 192,
                 channels: int = 512, scale: int = 8, **kwargs):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim

        # Block 1 — stem conv
        self.stem_conv = tf.keras.layers.Conv1D(
            channels, kernel_size=5, padding='same', use_bias=False)
        self.stem_bn = tf.keras.layers.BatchNormalization()

        # Blocks 2–5 — SE-Res2Blocks with increasing dilation
        self.block2 = SERes2Block(
            channels, scale=scale, dilation=2, name='block2')
        self.block3 = SERes2Block(
            channels, scale=scale, dilation=3, name='block3')
        self.block4 = SERes2Block(
            channels, scale=scale, dilation=4, name='block4')
        self.block5 = SERes2Block(
            channels, scale=scale, dilation=5, name='block5')

        # MFA — project concatenated multi-scale features
        # 4 blocks × 512 channels = 2048 → 1536
        self.mfa_conv = tf.keras.layers.Conv1D(
            1536, kernel_size=1, padding='same', use_bias=False)
        self.mfa_bn = tf.keras.layers.BatchNormalization()

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling()

        # Embedding projection
        self.fc = tf.keras.layers.Dense(embedding_dim, use_bias=False)
        self.emb_bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        """
        Args:
            x: (batch, 80, T) — channels-first spectrogram from features.py

        Returns:
            embedding: (batch, 192) — L2-normalized, ready for ArcFace loss
        """
        # Transpose to channels-last for Conv1D: (batch, T, 80)
        x = tf.transpose(x, perm=[0, 2, 1])

        # Block 1 — stem
        x = self.stem_conv(x, training=training)
        x = self.stem_bn(x,   training=training)
        x = tf.nn.relu(x)                          # (batch, T, 512)

        # Blocks 2–5 — keep all outputs for MFA
        h2 = self.block2(x,  training=training)     # (batch, T, 512)
        h3 = self.block3(h2, training=training)     # (batch, T, 512)
        h4 = self.block4(h3, training=training)     # (batch, T, 512)
        h5 = self.block5(h4, training=training)     # (batch, T, 512)

        # MFA — concatenate all block outputs
        mfa = tf.concat([h2, h3, h4, h5], axis=-1)  # (batch, T, 2048)
        mfa = self.mfa_conv(mfa, training=training)  # (batch, T, 1536)
        mfa = self.mfa_bn(mfa,   training=training)
        mfa = tf.nn.relu(mfa)

        # Attentive Statistics Pooling → fixed-dim vector
        pooled = self.asp(mfa, training=training)    # (batch, 3072)

        # Project to embedding space
        emb = self.fc(pooled,    training=training)  # (batch, 192)
        emb = self.emb_bn(emb,   training=training)

        # L2 normalize — embeddings live on a unit hypersphere
        # Required by ArcFace loss and cosine similarity scoring
        emb = tf.math.l2_normalize(emb, axis=-1)    # (batch, 192)

        return emb

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'embedding_dim': self.embedding_dim})
        return cfg
