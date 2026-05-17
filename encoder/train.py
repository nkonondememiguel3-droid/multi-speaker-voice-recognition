import os
import time
import math
import logging
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from collections import defaultdict

from encoder.multi_task import MultiTaskECAPA

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BALANCED BATCH SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

class BalancedBatchSampler:
    """
    Yields batches of (spectrogram, labels) where each batch contains
    exactly N_SPEAKERS_PER_BATCH speakers × N_UTTERANCES_PER_SPEAKER
    utterances.

    This is critical for ArcFace training: if a batch contains only a
    few speakers, the loss has very few negative pairs to learn from.
    With 32 speakers × 8 utterances = 256 samples per batch, every
    forward pass sees 32 different speaker identities, giving the
    SubcenterArcFace loss a rich set of inter-speaker contrasts.

    The sampler reads a manifest CSV with columns:
        path        — absolute path to the .npy spectrogram file
        speaker_id  — integer 0..N-1
        gender      — integer 0..1
        language    — integer 0..9
        emotion     — integer 0..6

    At each epoch the speaker list is shuffled, then for each speaker
    their utterances are shuffled and N_UTTERANCES_PER_SPEAKER are
    drawn. If a speaker has fewer utterances than needed, we sample
    with replacement.
    """

    def __init__(
        self,
        manifest_path: str,
        n_speakers_per_batch: int = 32,
        n_utterances_per_speaker: int = 8,
    ):
        self.n_spk = n_speakers_per_batch
        self.n_utt = n_utterances_per_speaker

        df = pd.read_csv(manifest_path)

        # Group utterance paths by speaker
        self.speaker_utterances: dict[int, list[dict]] = defaultdict(list)
        for _, row in df.iterrows():
            self.speaker_utterances[int(row['speaker_id'])].append({
                'path':       row['path'],
                'speaker_id': int(row['speaker_id']),
                'gender':     int(row['gender']),
                'language':   int(row['language']),
                'emotion':    int(row['emotion']),
            })

        self.speaker_ids = list(self.speaker_utterances.keys())
        self.n_total_speakers = len(self.speaker_ids)

        logger.info(
            f"Sampler: {self.n_total_speakers} speakers, "
            f"batch={n_speakers_per_batch}×{n_utterances_per_speaker}="
            f"{n_speakers_per_batch * n_utterances_per_speaker} samples"
        )

    def __iter__(self):
        """
        Yields one batch dict at a time:
            {
              'specs':       np.ndarray (batch, 80, T)
              'speaker_id':  np.ndarray (batch,)
              'gender':      np.ndarray (batch,)
              'language':    np.ndarray (batch,)
              'emotion':     np.ndarray (batch,)
            }
        """
        np.random.shuffle(self.speaker_ids)

        for i in range(0, self.n_total_speakers - self.n_spk + 1, self.n_spk):
            batch_speakers = self.speaker_ids[i:i + self.n_spk]
            specs, speaker_ids, genders, languages, emotions = [], [], [], [], []

            for spk_id in batch_speakers:
                utts = self.speaker_utterances[spk_id]

                # Sample with replacement if speaker has fewer utterances than needed
                replace = len(utts) < self.n_utt
                chosen = np.random.choice(
                    len(utts), size=self.n_utt, replace=replace)

                for idx in chosen:
                    utt = utts[idx]
                    spec = np.load(utt['path'])          # (80, T)
                    specs.append(spec)
                    speaker_ids.append(utt['speaker_id'])
                    genders.append(utt['gender'])
                    languages.append(utt['language'])
                    emotions.append(utt['emotion'])

            # Pad all specs in the batch to the same T
            specs = _pad_batch(specs)                    # (batch, 80, T_max)

            yield {
                'specs':      specs,
                'speaker_id': np.array(speaker_ids, dtype=np.int32),
                'gender':     np.array(genders,     dtype=np.int32),
                'language':   np.array(languages,   dtype=np.int32),
                'emotion':    np.array(emotions,     dtype=np.int32),
            }

    def steps_per_epoch(self) -> int:
        return self.n_total_speakers // self.n_spk


def _pad_batch(specs: list[np.ndarray]) -> np.ndarray:
    """
    Pad a list of (80, T_i) spectrograms to (B, 80, T_max) by
    zero-padding along the time axis.

    Spectrograms have variable T because voiced segments have different
    durations. The encoder handles this correctly — the attentive pooling
    layer aggregates over whatever T is present. Padding with zeros adds
    silence frames which the attention weights will naturally suppress.
    """
    T_max = max(s.shape[1] for s in specs)
    result = np.zeros((len(specs), 80, T_max), dtype=np.float32)
    for i, s in enumerate(specs):
        result[i, :, :s.shape[1]] = s
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. CYCLIC LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cyclical Learning Rate schedule — triangular2 policy.

    The learning rate oscillates between base_lr and max_lr over a
    cycle of 2 * step_size steps. In triangular2 mode, the max_lr is
    halved at the end of each full cycle, creating a decaying envelope.

    Why cyclic LR for speaker recognition?
    Speaker embeddings trained with ArcFace benefit from periodic high-LR
    phases that help escape sharp minima in angular space, followed by
    low-LR consolidation phases that tighten cluster boundaries.

    step_size is typically 2–8 × steps_per_epoch. Here we use 4×.

    Warmup: for the first `warmup_steps` steps, LR rises linearly from
    0 to base_lr before the cyclic schedule begins. This prevents large
    gradient updates from the randomly initialized ArcFace weight matrix
    from destabilizing the pre-trained backbone early in fine-tuning.
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        max_lr: float = 1e-3,
        step_size: int = 2000,
        warmup_steps: int = 500,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Linear warmup phase
        warmup_lr = self.base_lr * (step / self.warmup_steps)

        # Cyclic phase (triangular2)
        cycle = tf.floor(1.0 + step / (2.0 * self.step_size))
        x = tf.abs(step / self.step_size - 2.0 * cycle + 1.0)
        scale = 1.0 / (2.0 ** (cycle - 1.0))
        cyclic_lr = self.base_lr + \
            (self.max_lr - self.base_lr) * tf.maximum(0.0, 1.0 - x) * scale

        # Use warmup LR until warmup_steps, then cyclic
        return tf.where(step < self.warmup_steps, warmup_lr, cyclic_lr)

    def get_config(self):
        return {
            'base_lr':      self.base_lr,
            'max_lr':       self.max_lr,
            'step_size':    self.step_size,
            'warmup_steps': self.warmup_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. EER METRIC  (Equal Error Rate)
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(embeddings: np.ndarray, speaker_ids: np.ndarray) -> float:
    """
    Compute Equal Error Rate on a held-out set.

    EER is the threshold at which False Accept Rate == False Reject Rate.
    Lower is better; target for this project is < 5%.

    Method:
        1. Compute all pairwise cosine similarities between embeddings
        2. Label each pair: 1 if same speaker, 0 if different
        3. Sweep threshold from -1 to 1; find crossing point of FAR and FRR

    This is an O(N²) operation — only run on a small dev subset
    (e.g. 200 utterances, 20 speakers × 10 utterances each).
    """
    # Cosine similarity matrix: embeddings are already L2-normalized
    sim_matrix = embeddings @ embeddings.T          # (N, N)

    # Build same/different speaker labels
    N = len(speaker_ids)
    scores = []
    labels = []
    for i in range(N):
        for j in range(i + 1, N):
            scores.append(sim_matrix[i, j])
            labels.append(1 if speaker_ids[i] == speaker_ids[j] else 0)

    scores = np.array(scores)
    labels = np.array(labels)

    # Sweep thresholds
    thresholds = np.linspace(-1.0, 1.0, 200)
    min_diff = float('inf')
    eer = 1.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        # FAR: accepted impostors / total impostors
        # FRR: rejected genuine / total genuine
        imp_mask = labels == 0
        gen_mask = labels == 1
        far = np.mean(preds[imp_mask] == 1) if imp_mask.any() else 0.0
        frr = np.mean(preds[gen_mask] == 0) if gen_mask.any() else 0.0
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0

    return float(eer)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ONE TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────

@tf.function
def train_step(
    model:     MultiTaskECAPA,
    optimizer: tf.keras.optimizers.Optimizer,
    specs:     tf.Tensor,
    labels:    dict,
) -> dict:
    """
    Single gradient update.

    Mixed precision: the optimizer is wrapped with LossScaleOptimizer
    in main(), so gradients are computed in float16 and weights updated
    in float32 automatically. No changes needed here.

    Gradient clipping (max_norm=5.0) prevents exploding gradients during
    the early high-LR phase of the cyclic schedule. Without clipping,
    the first few batches after a LR peak can corrupt the ArcFace weights.

    @tf.function compiles this to a TF graph — roughly 3× faster than
    eager mode on CPU, and essential on GPU.
    """
    with tf.GradientTape() as tape:
        total_loss, loss_dict = model.compute_loss(
            specs, labels, training=True)

    gradients = tape.gradient(total_loss, model.trainable_variables)

    # Gradient clipping — operate on raw gradients before optimizer apply
    gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=5.0)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model: MultiTaskECAPA, checkpoint_dir: str,
                    epoch: int, eer: float):
    """Save model weights to a Keras weights file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = f"epoch_{epoch:03d}_eer_{eer:.4f}.weights.h5"   # ← add .weights.h5
    path = os.path.join(checkpoint_dir, tag)
    model.save_weights(path)
    logger.info(f"Checkpoint saved → {path}")
    return path


def load_checkpoint(model: MultiTaskECAPA, checkpoint_path: str):
    """Restore weights from a checkpoint produced by save_checkpoint."""
    model.load_weights(checkpoint_path)
    logger.info(f"Weights restored from {checkpoint_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────


def train(
    manifest_train:  str,
    manifest_dev:    str,
    checkpoint_dir:  str = "./checkpoints",
    n_speakers:      int = 80,
    embedding_dim:   int = 192,
    n_epochs:        int = 50,
    n_spk_per_batch: int = 32,
    n_utt_per_spk:   int = 8,
    base_lr:         float = 1e-4,
    max_lr:          float = 1e-3,
    warmup_steps:    int = 500,
    eer_eval_every:  int = 5,
    resume_from:     str = None,
):
    """
    Full training loop.

    Args:
        manifest_train:  path to CSV for training split
        manifest_dev:    path to CSV for dev/validation split
        checkpoint_dir:  directory to save checkpoints
        n_speakers:      total number of speakers in training set
        embedding_dim:   speaker embedding size (192)
        n_epochs:        number of full passes over the training speakers
        n_spk_per_batch: speakers per batch for balanced sampler
        n_utt_per_spk:   utterances per speaker per batch
        base_lr:         CyclicLR lower bound
        max_lr:          CyclicLR upper bound
        warmup_steps:    linear warmup before cyclic schedule begins
        eer_eval_every:  run EER evaluation every N epochs
        resume_from:     path to checkpoint to resume from (optional)
    """

    # ── Mixed precision ──────────────────────────────────────────────────────
    # Computes in float16, stores weights in float32.
    # Roughly 2× faster on GPU with Tensor Cores; no accuracy loss.
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logger.info("Mixed precision: float16 compute, float32 weights")

    # Model
    model = MultiTaskECAPA(n_speakers=n_speakers, embedding_dim=embedding_dim)

    # After — build each sub-component explicitly, no ambiguity
    # ── Build model ───────────────────────────────────────────────────────────
    dummy_specs = tf.zeros((2, 80, 300), dtype=tf.float16)
    dummy_labels = {
        'speaker_id': tf.constant([0, 0], dtype=tf.int32),
        'gender':     tf.constant([0, 0], dtype=tf.int32),
        'language':   tf.constant([0, 0], dtype=tf.int32),
        'emotion':    tf.constant([0, 0], dtype=tf.int32),
    }

    # Run the full forward pass — builds backbone, heads, and ArcFace
    _ = model.compute_loss(dummy_specs, dummy_labels, training=False)

    # Count params by summing trainable variables directly —
    # avoids Keras's built-state check which is unreliable for
    # models with non-standard call signatures (like ArcFace)
    total_params = sum(
        tf.size(v).numpy() for v in model.trainable_variables
    )
    logger.info(f"Model ready — {total_params:,} trainable parameters")

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Cap n_spk_per_batch to the actual number of speakers in the dataset
    sampler = BalancedBatchSampler(
        manifest_train,
        n_speakers_per_batch=min(n_spk_per_batch, n_speakers),
        n_utterances_per_speaker=n_utt_per_spk,
    )

    steps_per_epoch = sampler.steps_per_epoch()

    # Guard: abort early with a clear message rather than silently doing nothing
    if steps_per_epoch == 0:
        raise ValueError(
            f"steps_per_epoch=0: not enough speakers in the training manifest "
            f"({sampler.n_total_speakers} speakers, need at least n_spk_per_batch={sampler.n_spk}). "
            f"Add more speakers to your dataset or reduce n_spk_per_batch."
        )

    schedule = CyclicLR(
        base_lr=base_lr,
        max_lr=max_lr,
        step_size=4 * steps_per_epoch,   # one cycle = 8 epochs
        warmup_steps=warmup_steps,
    )
    # LossScaleOptimizer wraps AdamW for mixed precision training
    inner_opt = tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=2e-5,
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(inner_opt)

    # ── Dev set for EER ──────────────────────────────────────────────────────
    dev_specs, dev_ids = _load_dev_set(manifest_dev)

    # ── Training state ───────────────────────────────────────────────────────
    best_eer = float('inf')
    best_ckpt_path = None
    history = []   # list of per-epoch dicts for optional plotting

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        epoch_losses = defaultdict(list)

        logger.info(f"\n{'─'*60}")
        logger.info(f"Epoch {epoch}/{n_epochs}")
        logger.info(f"{'─'*60}")

        # ── Batch loop ───────────────────────────────────────────────────────
        for step, batch in enumerate(sampler, start=1):

            specs = tf.constant(batch['specs'], dtype=tf.float16)
            labels = {
                'speaker_id': tf.constant(batch['speaker_id']),
                'gender':     tf.constant(batch['gender']),
                'language':   tf.constant(batch['language']),
                'emotion':    tf.constant(batch['emotion']),
            }

            loss_dict = train_step(model, optimizer, specs, labels)

            for k, v in loss_dict.items():
                epoch_losses[k].append(float(v))

            # ── Step-level logging (every 10 steps) ──────────────────────────
            if step % 10 == 0 or step == steps_per_epoch:
                current_lr = float(schedule(optimizer.iterations))
                logger.info(
                    f"  step {step:>4}/{steps_per_epoch} | "
                    f"loss={epoch_losses['total'][-1]:.4f} | "
                    f"spk={epoch_losses['speaker'][-1]:.4f} | "
                    f"lr={current_lr:.2e}"
                )

        # ── Epoch summary ─────────────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        mean_loss = {k: float(np.mean(v)) for k, v in epoch_losses.items()}

        logger.info(
            f"Epoch {epoch} done in {elapsed:.0f}s | "
            f"mean_loss={mean_loss['total']:.4f} | "
            f"speaker={mean_loss['speaker']:.4f} | "
            f"gender={mean_loss['gender']:.4f} | "
            f"language={mean_loss['language']:.4f} | "
            f"emotion={mean_loss['emotion']:.4f}"
        )

        # ── EER evaluation ────────────────────────────────────────────────────
        eer = float('inf')
        if epoch % eer_eval_every == 0 or epoch == n_epochs:
            eer = _evaluate_eer(model, dev_specs, dev_ids)
            logger.info(f"EER @ epoch {epoch}: {eer * 100:.2f}%")

            if eer < best_eer:
                best_eer = eer
                best_ckpt_path = save_checkpoint(
                    model, checkpoint_dir, epoch, eer)
                logger.info(f"  ↳ New best EER: {best_eer * 100:.2f}%")

        # ── Save every 5 epochs regardless of EER ────────────────────────────
        if epoch % 5 == 0:
            save_checkpoint(model, checkpoint_dir, epoch, eer)

        history.append({
            'epoch':    epoch,
            'eer':      eer,
            **mean_loss,
        })

    logger.info(f"\nTraining complete.")
    logger.info(f"Best EER : {best_eer * 100:.2f}%")
    logger.info(f"Best ckpt: {best_ckpt_path}")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 7. DEV SET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_dev_set(
    manifest_dev: str,
    max_utterances: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a fixed dev set for EER evaluation.

    Loads up to max_utterances spectrograms, pads them to the same T,
    and returns them as a single array. 200 utterances across 20 speakers
    gives 19,900 pairs — enough for a reliable EER estimate without being
    slow.
    """
    df = pd.read_csv(manifest_dev).head(max_utterances)
    specs = []
    ids = []
    for _, row in df.iterrows():
        spec = np.load(row['path'])    # (80, T)
        specs.append(spec)
        ids.append(int(row['speaker_id']))

    specs_padded = _pad_batch(specs)   # (N, 80, T_max)
    return specs_padded, np.array(ids, dtype=np.int32)


def _evaluate_eer(
    model:    MultiTaskECAPA,
    specs:    np.ndarray,
    spk_ids:  np.ndarray,
) -> float:
    """
    Run encoder on dev specs and compute EER.
    Uses encode() — the inference path without auxiliary heads.
    """
    specs_tf = tf.constant(specs, dtype=tf.float16)
    embeddings = model.encode(specs_tf, training=False).numpy()   # (N, 192)
    return compute_eer(embeddings, spk_ids)
