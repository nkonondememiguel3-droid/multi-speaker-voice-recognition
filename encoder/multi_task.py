import tensorflow as tf
from encoder.ecapa_tdnn import ECAPA_TDNN
from encoder.loss import SubcenterArcFace


class MultiTaskECAPA(tf.keras.Model):
    """
    ECAPA-TDNN with auxiliary classification heads for multi-task learning.

    The shared ECAPA backbone learns a general speech representation.
    Three auxiliary heads regularize it during training by forcing the
    embedding to also carry information about gender, language, and emotion.

    This prevents the backbone from overfitting purely to speaker identity
    in the 80-speaker fine-tuning set, since it must simultaneously solve
    three other speech tasks.

    Combined loss:
        L = L_arcface
          + 0.1 * L_gender
          + 0.1 * L_language
          + 0.05 * L_emotion

    The weights (λ) reflect importance: speaker ID is the primary task,
    gender and language are strong regularizers, emotion is a weak hint.

    IMPORTANT: at inference time only the backbone is used.
    Call encode() instead of call() to get embeddings without head overhead.
    """

    # Label counts
    N_GENDERS = 2
    N_LANGUAGES = 10
    N_EMOTIONS = 7

    def __init__(self, n_speakers: int, embedding_dim: int = 192,
                 lambda_gender: float = 0.1, lambda_language: float = 0.1,
                 lambda_emotion: float = 0.05, **kwargs):
        super().__init__(**kwargs)

        self.lambda_gender = lambda_gender
        self.lambda_language = lambda_language
        self.lambda_emotion = lambda_emotion

        # ── Shared backbone ──────────────────────────────────────────────────
        self.backbone = ECAPA_TDNN(embedding_dim=embedding_dim)

        # ── Speaker head (ArcFace) ───────────────────────────────────────────
        self.arcface = SubcenterArcFace(
            n_speakers=n_speakers,
            embedding_dim=embedding_dim,
        )

        # ── Auxiliary heads ──────────────────────────────────────────────────
        # Heads operate on the pre-normalized embedding (before L2 norm)
        # so they receive richer gradient signal.
        # Each is a single Dense layer — deliberately shallow so they don't
        # dominate the shared representation.
        self.gender_head = tf.keras.layers.Dense(
            self.N_GENDERS,   name='gender_head')
        self.language_head = tf.keras.layers.Dense(
            self.N_LANGUAGES, name='language_head')
        self.emotion_head = tf.keras.layers.Dense(
            self.N_EMOTIONS,  name='emotion_head')

    def call(self, x, training=False):
        """
        Full forward pass for training.

        Args:
            x: (batch, 80, T) spectrogram

        Returns:
            embedding: (batch, 192) L2-normalized — used by ArcFace
            gender_logits:   (batch, 2)
            language_logits: (batch, 10)
            emotion_logits:  (batch, 7)
        """
        embedding = self.backbone(x, training=training)   # (batch, 192)

        gender_logits = self.gender_head(embedding,   training=training)
        language_logits = self.language_head(embedding, training=training)
        emotion_logits = self.emotion_head(embedding,  training=training)

        return embedding, gender_logits, language_logits, emotion_logits

    def encode(self, x, training=False):
        """
        Inference-only path. Returns only the L2-normalized embedding.
        Heads are not called — faster and what the backend expects.
        """
        return self.backbone(x, training=training)    # (batch, 192)

    def compute_loss(self, x, y, training=False):
        embedding, gender_logits, language_logits, emotion_logits = \
            self.call(x, training=training)

        # Primary: SubcenterArcFace — always float32 (pinned in loss.py)
        l_speaker = self.arcface(y['speaker_id'], embedding)

        # Auxiliary losses — cast to float32 explicitly.
        # The backbone runs in float16 under mixed precision, so logits
        # coming out of the auxiliary heads are float16. We must cast
        # before combining with the float32 ArcFace loss.
        l_gender = tf.cast(
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(y['gender'], tf.int32),
                    logits=tf.cast(gender_logits,   tf.float32),
                )
            ), tf.float32
        )
        l_language = tf.cast(
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(y['language'], tf.int32),
                    logits=tf.cast(language_logits, tf.float32),
                )
            ), tf.float32
        )
        l_emotion = tf.cast(
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(y['emotion'], tf.int32),
                    logits=tf.cast(emotion_logits,  tf.float32),
                )
            ), tf.float32
        )

        total_loss = (
            l_speaker
            + self.lambda_gender * l_gender
            + self.lambda_language * l_language
            + self.lambda_emotion * l_emotion
        )

        loss_dict = {
            'total':    total_loss,
            'speaker':  l_speaker,
            'gender':   l_gender,
            'language': l_language,
            'emotion':  l_emotion,
        }

        return total_loss, loss_dict

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'lambda_gender':   self.lambda_gender,
            'lambda_language': self.lambda_language,
            'lambda_emotion':  self.lambda_emotion,
        })
        return cfg
