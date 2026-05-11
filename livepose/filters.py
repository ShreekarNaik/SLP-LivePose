"""Temporal filters and post-processing for real-time pose smoothing.

PoseFilter          — One-Euro adaptive low-pass filter (J×3 scalars)
BoneLengthPrior     — learn bone lengths from early frames, then freeze
check_anthropometric_bounds — quick plausibility gate using known human ranges
fabrik_fit          — FABRIK IK to fit skeleton to learned bone lengths

References:
  Casiez et al. (2012). "1 € Filter." CHI '12.
  Welford, B.P. (1962). "Note on a method for calculating corrected sums."
  Aristidou & Lasenby (2011). "FABRIK: A fast, iterative solver for the
    Inverse Kinematics problem." Graphical Models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from livepose.core import COCO17_SKELETON


# ---------------------------------------------------------------------------
# One-Euro filter (scalar)
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """One-Euro filter for a single scalar signal."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x: float | None = None
        self._dx: float = 0.0
        self._last_t: float | None = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-10))

    def __call__(self, t: float, x: float) -> float:
        if self._last_t is None or self._x is None:
            self._x = x
            self._dx = 0.0
            self._last_t = t
            return x

        dt = max(t - self._last_t, 1e-10)
        self._last_t = t

        dx_raw = (x - self._x) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        self._dx = alpha_d * dx_raw + (1.0 - alpha_d) * self._dx

        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        alpha = self._alpha(dt, cutoff)
        self._x = alpha * x + (1.0 - alpha) * self._x
        return self._x

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._last_t = None


# ---------------------------------------------------------------------------
# Per-pose filter (17 joints × 3 axes)
# ---------------------------------------------------------------------------

class PoseFilter:
    """One-Euro filter per joint per axis applied to triangulated 3D positions."""

    NUM_JOINTS = 17
    NUM_AXES = 3

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: list[list[OneEuroFilter]] = [
            [OneEuroFilter(min_cutoff, beta) for _ in range(self.NUM_AXES)]
            for _ in range(self.NUM_JOINTS)
        ]
        self._last_valid = [False] * self.NUM_JOINTS

    def filter(
        self,
        timestamp: float,
        points_3d: np.ndarray,  # (J, 3)
        valid_3d: np.ndarray,   # (J,) bool
    ) -> tuple[np.ndarray, np.ndarray]:
        result = points_3d.copy()
        for j in range(self.NUM_JOINTS):
            is_valid = bool(valid_3d[j])
            was_valid = self._last_valid[j]

            if not is_valid:
                if was_valid:
                    for ax in range(self.NUM_AXES):
                        self._filters[j][ax].reset()
                continue

            if not was_valid:
                for ax in range(self.NUM_AXES):
                    self._filters[j][ax].reset()

            for ax in range(self.NUM_AXES):
                result[j, ax] = self._filters[j][ax](timestamp, float(points_3d[j, ax]))

        self._last_valid = [bool(valid_3d[j]) for j in range(self.NUM_JOINTS)]
        return result, valid_3d

    def reset(self) -> None:
        for joint_filters in self._filters:
            for f in joint_filters:
                f.reset()
        self._last_valid = [False] * self.NUM_JOINTS

    def update_params(self, min_cutoff: float, beta: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        for joint_filters in self._filters:
            for f in joint_filters:
                f.min_cutoff = min_cutoff
                f.beta = beta


# ---------------------------------------------------------------------------
# B1: Anthropometric hard bounds
# ---------------------------------------------------------------------------

# Min/max bone lengths in millimetres from biomechanics literature.
# Covers a very wide range of human body sizes (children → large adults).
COCO17_BONE_BOUNDS_MM: dict[tuple[int, int], tuple[float, float]] = {
    (5, 7):  (200, 400),   # L shoulder-elbow (upper arm)
    (7, 9):  (180, 350),   # L elbow-wrist (forearm)
    (6, 8):  (200, 400),   # R shoulder-elbow
    (8, 10): (180, 350),   # R elbow-wrist
    (11, 13): (350, 550),  # L hip-knee (femur)
    (13, 15): (300, 500),  # L knee-ankle (tibia)
    (12, 14): (350, 550),  # R hip-knee
    (14, 16): (300, 500),  # R knee-ankle
    (5, 6):  (280, 500),   # shoulder width
    (11, 12): (200, 380),  # hip width
    (5, 11): (400, 650),   # L torso (shoulder-hip)
    (6, 12): (400, 650),   # R torso
}


def check_anthropometric_bounds(
    points_3d: np.ndarray,  # (J, 3)
    valid_3d: np.ndarray,   # (J,) bool
) -> float:
    """Return fraction of bones within anthropometric bounds (0.0–1.0).

    Bones where either endpoint is invalid are skipped (not counted).
    Returns 1.0 if no bones can be checked.
    """
    n_checked = 0
    n_ok = 0
    for (a, b), (lo, hi) in COCO17_BONE_BOUNDS_MM.items():
        if not (valid_3d[a] and valid_3d[b]):
            continue
        length = float(np.linalg.norm(points_3d[a] - points_3d[b]))
        n_checked += 1
        if lo <= length <= hi:
            n_ok += 1
    return n_ok / n_checked if n_checked > 0 else 1.0


# ---------------------------------------------------------------------------
# B2: Learn-then-freeze bone lengths
# ---------------------------------------------------------------------------

class BoneLengthPrior:
    """Learns bone lengths from early clean frames, then freezes.

    Uses Welford's online algorithm until all bones reach warmup_frames
    observations, then freezes the mean lengths for FABRIK.
    """

    def __init__(self, warmup_frames: int = 15) -> None:
        self._skeleton = list(COCO17_SKELETON)
        self.warmup_frames = warmup_frames

        n = len(self._skeleton)
        self._count = np.zeros(n, dtype=np.int64)
        self._mean = np.zeros(n, dtype=np.float64)
        self._M2 = np.zeros(n, dtype=np.float64)
        self._frozen = False
        self._frozen_lengths: np.ndarray | None = None  # (N_bones,)

    def observe(self, points_3d: np.ndarray, valid_3d: np.ndarray) -> None:
        """Accumulate statistics until warmup, then freeze. No-op after freeze."""
        if self._frozen:
            return
        for i, (a, b) in enumerate(self._skeleton):
            if not (valid_3d[a] and valid_3d[b]):
                continue
            length = float(np.linalg.norm(points_3d[a] - points_3d[b]))
            if length < 1.0:
                continue
            self._count[i] += 1
            delta = length - self._mean[i]
            self._mean[i] += delta / self._count[i]
            self._M2[i] += delta * (length - self._mean[i])

        # Freeze when ≥75% of bones have enough samples (face bones may never appear)
        observed = (self._count >= self.warmup_frames).sum()
        if observed >= max(1, int(0.75 * len(self._skeleton))):
            # Fill unobserved bones with the overall mean (fallback)
            mean_all = float(self._mean[self._count > 0].mean()) if (self._count > 0).any() else 200.0
            lengths = self._mean.copy()
            lengths[self._count < self.warmup_frames] = mean_all
            self._frozen = True
            self._frozen_lengths = lengths

    @property
    def is_ready(self) -> bool:
        return self._frozen

    @property
    def lengths(self) -> np.ndarray | None:
        """Frozen bone lengths (N_bones,) or None if not ready."""
        return self._frozen_lengths

    def reset(self) -> None:
        self._count[:] = 0
        self._mean[:] = 0.0
        self._M2[:] = 0.0
        self._frozen = False
        self._frozen_lengths = None

    @property
    def learned_lengths(self) -> dict[tuple[int, int], float]:
        """Bone → learned mean length (mm). Empty until frozen."""
        if not self._frozen or self._frozen_lengths is None:
            return {}
        return {
            bone: float(self._frozen_lengths[i])
            for i, bone in enumerate(self._skeleton)
        }


# ---------------------------------------------------------------------------
# B3: Progressive confidence-weighted bone model (replaces BoneLengthPrior)
# ---------------------------------------------------------------------------

@dataclass
class BoneStat:
    mean: float = 0.0
    variance: float = 0.0
    confidence: float = 0.0
    sample_count: int = 0


class BoneModel:
    """Progressive confidence-weighted bone length model.

    Never freezes — compliance decreases smoothly as observations accumulate.
    Replaces BoneLengthPrior + fabrik_fit with a soft XPBD constraint approach.
    """

    _INIT_COMPLIANCE   = 0.8   # compliance when bone has few samples
    _STAB_COMPLIANCE   = 0.4   # compliance during stabilization
    _LOCKED_COMPLIANCE = 0.1   # compliance once well-observed
    _STAB_SAMPLES      = 10    # per-bone count to leave init phase
    _LOCKED_SAMPLES    = 30    # per-bone count to enter locked phase
    _BASE_ALPHA        = 0.3
    _DECAY             = 0.05
    _REPROJ_THRESHOLD_PX  = 15.0
    _CORRUPTION_SIGMA     = 3.0
    _CORRUPTION_FRACTION  = 0.5

    _TORSO_BONES = frozenset({(5, 6), (5, 11), (6, 12), (11, 12)})
    _FACE_BONES  = frozenset({(0, 1), (0, 2), (1, 3), (2, 4)})

    def __init__(self) -> None:
        self._skeleton = list(COCO17_SKELETON)
        self._stats: list[BoneStat] = [BoneStat() for _ in range(len(self._skeleton))]
        self._stiffness: np.ndarray = self._compute_stiffness()

    def _compute_stiffness(self) -> np.ndarray:
        """Per-bone stiffness: torso 1.0×, limbs 0.7×, face/clavicle 0.3×."""
        stiff = np.empty(len(self._skeleton))
        for i, (a, b) in enumerate(self._skeleton):
            key = (min(a, b), max(a, b))
            if key in self._TORSO_BONES:
                stiff[i] = 1.0
            elif key in self._FACE_BONES:
                stiff[i] = 0.3
            else:
                stiff[i] = 0.7
        return stiff

    def get_compliance(self, bone_idx: int) -> float:
        """Compliance in [0, 1]: 1 = fully soft (ignored), 0 = fully rigid."""
        n = self._stats[bone_idx].sample_count
        if n < self._STAB_SAMPLES:
            return self._INIT_COMPLIANCE
        if n < self._LOCKED_SAMPLES:
            return self._STAB_COMPLIANCE
        return self._LOCKED_COMPLIANCE

    def target_lengths(self) -> np.ndarray:
        """Current best-estimate bone lengths (N_bones,). 0.0 where unobserved."""
        return np.array([s.mean for s in self._stats], dtype=np.float64)

    @property
    def phase(self) -> str:
        """Current phase: 'init', 'stabilizing', or 'locked'."""
        observed = [s.sample_count for s in self._stats if s.sample_count > 0]
        if not observed:
            return "init"
        median = sorted(observed)[len(observed) // 2]
        if median < self._STAB_SAMPLES:
            return "init"
        if median < self._LOCKED_SAMPLES:
            return "stabilizing"
        return "locked"

    def observe(
        self,
        points_3d: np.ndarray,      # (J, 3)
        valid_3d: np.ndarray,       # (J,) bool
        reproj_errors: np.ndarray,  # (J,) float — per-joint mean reproj error px
    ) -> None:
        """Update bone model from a new frame's observations."""
        # --- Corruption detection: skip frame if >50% of measurable bones
        #     deviate more than 3σ from the running model.
        deviations: list[float] = []
        for i, (a, b) in enumerate(self._skeleton):
            stat = self._stats[i]
            if stat.sample_count < 2:
                continue
            if not (valid_3d[a] and valid_3d[b]):
                continue
            length = float(np.linalg.norm(points_3d[a] - points_3d[b]))
            std = math.sqrt(max(stat.variance, 1e-10))
            deviations.append(abs(length - stat.mean) / std)

        if deviations:
            n_corrupt = sum(1 for d in deviations if d > self._CORRUPTION_SIGMA)
            if n_corrupt / len(deviations) > self._CORRUPTION_FRACTION:
                return  # corrupted frame — skip all updates

        # --- Per-bone update (confidence-gated) ---
        for i, (a, b) in enumerate(self._skeleton):
            if not (valid_3d[a] and valid_3d[b]):
                continue
            if (reproj_errors[a] > self._REPROJ_THRESHOLD_PX or
                    reproj_errors[b] > self._REPROJ_THRESHOLD_PX):
                continue

            length = float(np.linalg.norm(points_3d[a] - points_3d[b]))

            key = (min(a, b), max(a, b))
            if key in COCO17_BONE_BOUNDS_MM:
                lo, hi = COCO17_BONE_BOUNDS_MM[key]
                if not (lo <= length <= hi):
                    continue
            elif length < 1.0:
                continue

            stat = self._stats[i]
            n = stat.sample_count

            # Gate: penalise measurements that are far from the running mean
            if n >= 2:
                std = math.sqrt(max(stat.variance, 1e-10))
                gate_conf = math.exp(-0.5 * ((length - stat.mean) / (std * 3)) ** 2)
            else:
                gate_conf = 1.0

            alpha = self._BASE_ALPHA * gate_conf / (1.0 + n * self._DECAY)
            alpha = max(0.01, min(alpha, 1.0))

            if n == 0:
                stat.mean = length
                stat.variance = 0.0
            else:
                old_mean = stat.mean
                stat.mean = (1.0 - alpha) * old_mean + alpha * length
                stat.variance = (1.0 - alpha) * (
                    stat.variance + alpha * (length - old_mean) ** 2
                )
            stat.confidence = min(1.0, stat.confidence + 0.05)
            stat.sample_count += 1

    def reset(self) -> None:
        """Return to fresh state — high compliance, no observations."""
        self._stats = [BoneStat() for _ in range(len(self._skeleton))]


# ---------------------------------------------------------------------------
# B4: XPBD constraint solver (replaces fabrik_fit)
# ---------------------------------------------------------------------------

def xpbd_solve(
    points_3d: np.ndarray,    # (J, 3)
    valid_3d: np.ndarray,     # (J,) bool
    bone_model: BoneModel,
    skeleton: list[tuple[int, int]] | None = None,
    iterations: int = 5,
) -> np.ndarray:
    """XPBD-style position-based bone-length constraint solver.

    Applies soft bone-length constraints weighted by per-bone compliance.
    High compliance (early learning) ≈ barely moves joints.
    Low compliance (well-observed) ≈ enforces lengths more rigidly.
    Valid joints are soft targets; invalid joints are placed freely by constraint
    propagation from their valid neighbours.
    """
    if skeleton is None:
        skeleton = list(COCO17_SKELETON)

    result = points_3d.copy()
    targets = bone_model.target_lengths()

    for _ in range(iterations):
        for i, (a, b) in enumerate(skeleton):
            target = targets[i]
            if target < 1.0:  # not yet estimated
                continue

            compliance = bone_model.get_compliance(i)
            stiffness = bone_model._stiffness[i]
            eff = max(0.0, min(1.0, compliance / stiffness))
            weight = 1.0 - eff  # 0 = no correction, 1 = full correction

            pa = result[a]
            pb = result[b]
            vec = pa - pb
            dist = float(np.linalg.norm(vec))
            if not math.isfinite(dist) or dist < 1e-9:
                continue

            direction = vec / dist
            delta = (dist - target) * direction

            a_valid = bool(valid_3d[a])
            b_valid = bool(valid_3d[b])

            # Skip constraint if joint positions contain NaN (invalid joints
            # are filled with NaN in pts_3d; NaN propagates through norm/direction
            # and would corrupt valid neighbours).
            a_finite = bool(np.isfinite(result[a]).all())
            b_finite = bool(np.isfinite(result[b]).all())

            if a_valid and b_valid:
                # Both triangulated — apply soft split correction
                if not (a_finite and b_finite):
                    continue
                result[a] -= 0.5 * weight * delta
                result[b] += 0.5 * weight * delta
            elif a_valid and not b_valid:
                # b is free — place it at the correct distance from a
                if not a_finite:
                    continue
                result[b] = result[a] - direction * target
            elif b_valid and not a_valid:
                # a is free — place it at the correct distance from b
                if not b_finite:
                    continue
                result[a] = result[b] + direction * target
            # else: both invalid — skip, nothing useful to do

    return result


# ---------------------------------------------------------------------------
# B5: FABRIK IK fitting (legacy — superseded by xpbd_solve)
# ---------------------------------------------------------------------------

# COCO-17 kinematic chains for FABRIK.
# Each chain: list of joint indices from root outward.
# Root is the midpoint of hips (joints 11, 12) — a virtual joint.
# We define chains starting from actual joints, anchored at the root.
_FABRIK_CHAINS: list[list[int]] = [
    [11, 13, 15],   # L hip → L knee → L ankle
    [12, 14, 16],   # R hip → R knee → R ankle
    [11, 5, 7, 9],  # L hip → L shoulder → L elbow → L wrist
    [12, 6, 8, 10], # R hip → R shoulder → R elbow → R wrist
    [11, 5, 0],     # L hip → L shoulder → nose (head proxy)
]

# Bone lengths needed per chain: bone[i] connects chain[i] → chain[i+1]
# We look them up in COCO17_SKELETON by (min, max) canonical form.
def _bone_key(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def _build_chain_bone_indices(
    skeleton: list[tuple[int, int]],
    chain: list[int],
) -> list[int | None]:
    """Return indices into skeleton for each consecutive pair in chain."""
    skel_map = {_bone_key(a, b): i for i, (a, b) in enumerate(skeleton)}
    indices = []
    for i in range(len(chain) - 1):
        key = _bone_key(chain[i], chain[i + 1])
        indices.append(skel_map.get(key))
    return indices


def fabrik_fit(
    points_3d: np.ndarray,   # (J, 3)
    valid_3d: np.ndarray,    # (J,) bool
    bone_lengths: np.ndarray,  # (N_bones,) from BoneLengthPrior
    skeleton: list[tuple[int, int]] | None = None,
    iterations: int = 5,
) -> np.ndarray:
    """Fit skeleton to learned bone lengths using FABRIK per kinematic chain.

    Valid joints are treated as soft targets (pulled toward their triangulated
    positions after each FABRIK pass).  Invalid joints are placed freely by IK.
    Returns a modified copy of points_3d.
    """
    if skeleton is None:
        skeleton = list(COCO17_SKELETON)

    result = points_3d.copy()

    for chain in _FABRIK_CHAINS:
        # Get bone length indices for this chain
        bone_idxs = _build_chain_bone_indices(skeleton, chain)

        # Gather chain lengths — skip chains with missing bones
        lengths: list[float] = []
        skip = False
        for idx in bone_idxs:
            if idx is None or idx >= len(bone_lengths):
                skip = True
                break
            lengths.append(float(bone_lengths[idx]))
        if skip or not lengths:
            continue

        # Extract current joint positions for this chain
        pts = result[chain].copy()  # (len(chain), 3)

        # Pin root: use current position of first joint as anchor
        root = pts[0].copy()

        for _ in range(iterations):
            # Forward pass: end-effector toward root direction, fix lengths
            for i in range(len(chain) - 1, 0, -1):
                vec = pts[i - 1] - pts[i]
                norm = float(np.linalg.norm(vec))
                if norm < 1e-9:
                    continue
                pts[i - 1] = pts[i] + (vec / norm) * lengths[i - 1]

            # Backward pass: from root outward
            pts[0] = root
            for i in range(len(chain) - 1):
                vec = pts[i + 1] - pts[i]
                norm = float(np.linalg.norm(vec))
                if norm < 1e-9:
                    continue
                pts[i + 1] = pts[i] + (vec / norm) * lengths[i]

        # Write back only joints that were valid (preserve their general position)
        # For invalid joints, IK placement is the best we can do
        for local_i, joint_idx in enumerate(chain):
            if valid_3d[joint_idx]:
                # Soft pull: blend IK result with triangulated target
                result[joint_idx] = 0.5 * pts[local_i] + 0.5 * points_3d[joint_idx]
            else:
                result[joint_idx] = pts[local_i]

    return result
