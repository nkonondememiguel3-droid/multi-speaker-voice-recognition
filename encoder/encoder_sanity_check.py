import numpy as np
import tensorflow as tf
from encoder.ecapa_tdnn import ECAPA_TDNN
from encoder.multi_task import MultiTaskECAPA

# Simulate a batch of 4 spectrograms of varying length padded to T=300
batch = tf.constant(np.random.randn(4, 80, 300).astype('float32'))

# ── Test backbone alone ────────────────────────────────────────────────────
backbone = ECAPA_TDNN(embedding_dim=192)
embeddings = backbone(batch, training=False)

print(f"Embedding shape : {embeddings.shape}")       # (4, 192)
print(f"Embedding norms : {tf.norm(embeddings, axis=-1).numpy()}")  # all ~1.0
print(f"Parameter count : {backbone.count_params():,}")             # ~6.5M

# ── Test multi-task model ──────────────────────────────────────────────────
model = MultiTaskECAPA(n_speakers=80)
labels = {
    'speaker_id': tf.constant([0, 1, 2, 3]),
    'gender':     tf.constant([0, 1, 0, 1]),
    'language':   tf.constant([0, 0, 1, 2]),
    'emotion':    tf.constant([0, 1, 2, 3]),
}

total_loss, loss_dict = model.compute_loss(batch, labels, training=True)

print(f"\nTotal loss : {total_loss.numpy():.4f}")
for k, v in loss_dict.items():
    print(f"  {k:10s}: {v.numpy():.4f}")

# Inference path (no heads)
emb = model.encode(batch, training=False)
print(f"\nInference embedding shape : {emb.shape}")   # (4, 192)
