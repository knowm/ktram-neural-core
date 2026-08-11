"""Classifier — the L1 supervised recoder (the RankCut / LinearClassifier routine) in torch.

One neural lane per output channel over a bank of differential-pair address spaces, as a
``torch.nn.Module``: AATs in, AATs out, the analog contained inside. ``Ga``/``Gb`` are
registered integer buffers — never Parameters, no autograd anywhere — so ``.to(device)`` and
``state_dict()`` come for free and the state dict is the live-state twin of the frozen pack.

The two verbs, exactly the oracle's:

    read(aat)          -> out_aat   non-disturbing sub-threshold read, rank-cut recoded
    adapt(aat, target) -> out_aat   adapting FF read + supervised RH/RL/RF, recoded

plus the opt-in ``read_sampled(aat, T, generator)`` noisy read. An AAT batch is ``[..., k]``
int64, symbols in [0, S), -1 = NONE; leading batch axes carry through untouched. ``target``
is an output-space AAT (lane indices, -1-padded). The decoder-style many-heads case (784
independent per-pixel classifiers in one Core) is this same module: lanes are lanes, and a
784-wide target AAT drives RH on each pixel's true lane — per-lane routines are identical
and lanes are independent, so the flat bank IS the stacked classifiers.

``num_groups=G`` stacks G such classifiers as one module (weights ``[G, L, K, S]``, the
BasisEncoder shape): reads score every group, targets grow a group axis (``[..., G, t]``
lane indices within each group), rank-cut recodes per group. With ``per_group=True`` the
AAT is ``[..., G, k]`` and group g reads/teaches on its OWN AAT — the completion-bank
self-echo fix: hand group g a copy of the context with its own slot set to NONE, and one
batched call reads or teaches every group with its structural hole. Groups are disjoint
lanes, so grouped == G independent flat Classifiers, byte for byte.

Batching semantics (spec 08 §6): lane-vectorization is exact; the leading batch is ONE adapt
step, reads stale within it. B = 1 is bit-for-bit the oracle's serial sequence — one FF-phase
update then one feedback-phase update per synapse, clamps in oracle order (the FF clamp lands
before the RF re-read). Byte model only: a float/RS/MSS lift raises rather than approximating.
"""

import numpy as np
import torch
from torch import nn

from . import _lane
from ._lane import GMAX, GMIN, NoiseParams, check_feedback, java_round, rank_cut

# Byte-model Core control-parameter defaults (core.MODEL_DEFAULTS["byte"]).
FORWARD_V = 1.0
REVERSE_V = -1.0
FORWARD_LOW_V = 0.05


class Classifier(nn.Module):
    """A bank of ``num_lanes`` neural lanes over ``num_spaces`` spaces of ``num_channels``
    channels each, with the (Vt, N) rank-cut readout.

    Two weight tiers, one module. Live (trainable): ``ga``/``gb`` int32 buffers in [1, 255],
    built here (oracle-matched seeded init) or lifted by ``from_core``. Frozen (inference /
    interchange): int8 (diff, mag) with global scales, loaded by ``from_pack`` — reads work,
    ``adapt`` raises. ``to_pack`` writes the same export format back.
    """

    def __init__(self, num_lanes, num_spaces, num_channels, *, num_groups=None,
                 Vt=0.0, N=None, init="medium", seed=None,
                 forward_voltage=FORWARD_V, reverse_voltage=REVERSE_V,
                 forward_low_voltage=FORWARD_LOW_V, noise=None):
        super().__init__()
        self.num_lanes = num_lanes                # per-group lanes when grouped
        self.num_spaces = num_spaces
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.Vt = Vt
        self.N = N
        self.vf = float(forward_voltage)
        self.vr = float(reverse_voltage)
        self.flv = float(forward_low_voltage)
        # The sub-threshold read is non-disturbing because its byte update rounds to zero
        # (|flv|*(1+|y|) < 0.5). The fast read path banks on that; a low voltage that breaks
        # it would need the oracle's update-on-read, so refuse it here.
        if java_round(torch.tensor(2.0 * abs(self.flv))).item() != 0.0:
            raise ValueError(
                f"forward_low_voltage {self.flv} would disturb a byte read; keep |flv| < 0.25")
        self.noise = noise if noise is not None else NoiseParams(v_read=abs(self.flv))
        self.frozen = False
        # CPU reads run as an embedding bag over a derived table (the 08d race winner),
        # cached per read mode (per_group needs a different row layout than the shared
        # read) and rebuilt lazily after any adapt. Anyone mutating ga/gb by hand must
        # clear it (`self._bag = {}`).
        self._bag = {}
        if num_groups is None:
            ga, gb = _lane.init_weights((num_lanes, num_spaces, num_channels), init, seed)
        else:
            # Group g draws from seed + g (the BasisEncoder rule); a list gives explicit
            # per-group seeds.
            if seed is None:
                seeds = [None] * num_groups
            elif isinstance(seed, (list, tuple)):
                if len(seed) != num_groups:
                    raise ValueError(f"{len(seed)} seeds for {num_groups} groups")
                seeds = list(seed)
            else:
                seeds = [seed + g for g in range(num_groups)]
            parts = [_lane.init_weights((num_lanes, num_spaces, num_channels), init, s)
                     for s in seeds]
            ga = torch.stack([p[0] for p in parts])
            gb = torch.stack([p[1] for p in parts])
        self.register_buffer("ga", ga)
        self.register_buffer("gb", gb)

    # ----- lifting a trained oracle Core (the live tier) -----

    @classmethod
    def from_core(cls, core, *, Vt=0.0, N=None):
        """Lift a trained oracle Core (or a RankCut / LinearClassifier wrapping one) into a
        torch Classifier — same integer state, same voltages, same noise coefficients.

        A list/tuple of cores (or RankCuts) lifts into ONE grouped module, group g = element
        g — every element must share dims, voltages, and readout shape."""
        if isinstance(core, (list, tuple)):
            return cls._from_cores(list(core), Vt=Vt, N=N)
        if hasattr(core, "Vt"):                      # a RankCut carries its readout shape
            Vt, N = core.Vt, core.N
        core = getattr(core, "core", core)
        if core.model_name != "byte":
            raise NotImplementedError(
                f"torch backend implements the byte model only, got {core.model_name!r}")
        L = core.num_lanes
        K = core.spaces_per_lane
        S = core.crossbar_rows * core.crossbar_cols
        self = cls(L, K, S, Vt=Vt, N=N, init="low_noiseless", seed=None,
                   forward_voltage=core.forward_voltage, reverse_voltage=core.reverse_voltage,
                   forward_low_voltage=core.forward_low_voltage,
                   noise=NoiseParams.from_core(core))
        ga, gb = cls._lift_weights(core, L, K, S)
        self.ga.copy_(torch.from_numpy(ga))
        self.gb.copy_(torch.from_numpy(gb))
        return self

    @staticmethod
    def _lift_weights(core, L, K, S):
        ga = np.empty((L, K, S), dtype=np.int32)
        gb = np.empty((L, K, S), dtype=np.int32)
        for lane_i in range(L):
            lane = core.lane(lane_i)
            for k, space in enumerate(lane.spaces):
                ga[lane_i, k] = [int(space.a.device_at(s).g()) for s in range(S)]
                gb[lane_i, k] = [int(space.b.device_at(s).g()) for s in range(S)]
        return ga, gb

    @classmethod
    def _from_cores(cls, sources, *, Vt, N):
        if hasattr(sources[0], "Vt"):
            Vt, N = sources[0].Vt, sources[0].N
            if any(getattr(s, "Vt", Vt) != Vt or getattr(s, "N", N) != N for s in sources):
                raise ValueError("grouped lift needs one (Vt, N) shared by every group")
        cores = [getattr(s, "core", s) for s in sources]
        c0 = cores[0]
        if any(c.model_name != "byte" for c in cores):
            raise NotImplementedError("torch backend implements the byte model only")
        L = c0.num_lanes
        K = c0.spaces_per_lane
        S = c0.crossbar_rows * c0.crossbar_cols
        for c in cores[1:]:
            if (c.num_lanes, c.spaces_per_lane, c.crossbar_rows * c.crossbar_cols,
                    c.forward_voltage, c.reverse_voltage, c.forward_low_voltage) != \
                    (L, K, S, c0.forward_voltage, c0.reverse_voltage, c0.forward_low_voltage):
                raise ValueError("grouped lift needs identical dims and voltages per group")
        self = cls(L, K, S, num_groups=len(cores), Vt=Vt, N=N, init="low_noiseless",
                   seed=None, forward_voltage=c0.forward_voltage,
                   reverse_voltage=c0.reverse_voltage,
                   forward_low_voltage=c0.forward_low_voltage,
                   noise=NoiseParams.from_core(c0))
        for g, c in enumerate(cores):
            ga, gb = cls._lift_weights(c, L, K, S)
            self.ga[g].copy_(torch.from_numpy(ga))
            self.gb[g].copy_(torch.from_numpy(gb))
        return self

    # ----- the frozen pack tier -----

    @classmethod
    def from_pack(cls, path, section="classifier", *, Vt=0.0, N=None):
        """Load one section ("classifier" or "decoder") of a frozen generator-weights export.
        The module reads from the quantized (diff, mag) exactly as the export's own validation
        does; ``adapt`` raises (integer Ga/Gb are not recoverable from the quantized pack)."""
        from .pack import load_section
        qdiff, qmag, diff_scale, mag_scale, noise = load_section(path, section)
        if qdiff.ndim == 4:                        # a grouped bank packs as [G, L, K, S]
            G, L, K, S = qdiff.shape
        else:
            G = None
            L, K, S = qdiff.shape
        self = cls(L, K, S, num_groups=G, Vt=Vt, N=N, init="low_noiseless",
                   noise=NoiseParams(**{k: v for k, v in noise.items() if k != "v_fflv"},
                                     v_read=noise["v_fflv"]) if noise else None)
        self.frozen = True
        delattr(self, "ga")                       # the live tier does not exist in a pack
        delattr(self, "gb")
        self.register_buffer("qdiff", torch.from_numpy(qdiff))
        self.register_buffer("qmag", torch.from_numpy(qmag))
        self.diff_scale = float(diff_scale)
        self.mag_scale = float(mag_scale)
        return self

    def to_pack(self, path, section="classifier"):
        """Quantize and write this module's weights into a generator-weights ``.npz`` pack
        (creating or updating the file's section in place). The layout is the existing export
        format — nothing new."""
        from .pack import save_section
        save_section(path, section, *self._diff_mag_float(), self.noise)

    def _diff_mag_float(self):
        if self.frozen:
            return (self.qdiff.numpy().astype(np.float32) * self.diff_scale,
                    self.qmag.numpy().astype(np.float32) * self.mag_scale)
        ga = self.ga.cpu().numpy().astype(np.float32)
        gb = self.gb.cpu().numpy().astype(np.float32)
        return ga - gb, ga + gb

    # ----- reading -----

    def _load_from_state_dict(self, *args, **kwargs):
        self._bag = {}                        # loaded weights invalidate the read tables
        super()._load_from_state_dict(*args, **kwargs)

    def _check_per_group(self, per_group):
        if per_group and self.num_groups is None:
            raise ValueError("per_group=True needs a grouped Classifier (num_groups=G)")

    def _read_sums(self, aat, per_group=False, fresh=False):
        """The (top, bottom) sums: embedding-bag table on CPU reads, the direct gather
        elsewhere and for the fresh reads inside adapt (the table would be stale)."""
        self._check_per_group(per_group)
        w1, w2 = (self.qdiff, self.qmag) if self.frozen else (self.ga, self.gb)
        if not fresh and aat.device.type == "cpu" and w1.device.type == "cpu":
            key = bool(per_group)
            if key not in self._bag:
                if self.frozen and _lane.has_fused_op():
                    self._bag[key] = _lane.fused_table(self.qdiff, self.qmag,
                                                       per_group=per_group)
                else:
                    self._bag[key] = _lane.bag_table(w1, w2, paired=not self.frozen,
                                                     per_group=per_group)
            out_shape = None if self.num_groups is None else \
                (self.num_groups, self.num_lanes)
            return _lane.bag_read_sums(self._bag[key], aat,
                                       self.num_spaces, self.num_channels,
                                       G=self.num_groups if per_group else None,
                                       out_shape=out_shape)
        return _lane.read_sums(w1, w2, aat, per_group=per_group, paired=not self.frozen)

    def _y_m(self, aat, per_group=False, fresh=False):
        """Clean y and total active magnitude m (real units) per lane: ``[..., L]`` each
        (``[..., G, L]`` grouped)."""
        top, bot = self._read_sums(aat, per_group, fresh)
        dt = _lane.y_dtype(top.device)
        if self.frozen:
            y = _lane.divide(top, bot, dt, self.diff_scale, self.mag_scale)
            m = bot.to(dt) * self.mag_scale
        else:
            y = _lane.divide(top, bot, dt)
            m = bot.to(dt)
        return y, m

    def read_y(self, aat, per_group=False):
        """The sub-threshold y vector, ``[..., L]`` (``[..., G, L]`` grouped) —
        debug/instrumentation, not the interface."""
        return self._y_m(aat, per_group)[0]

    def read(self, aat, per_group=False):
        """Inference: non-disturbing read on every lane, rank-cut recoded to ``[..., N]``
        (``[..., G, N]`` grouped, each group cut over its own lanes)."""
        return rank_cut(self.read_y(aat, per_group), self.Vt, self.N)

    def read_sampled(self, aat, T, generator=None, per_group=False):
        """Opt-in noisy read at temperature T (the read_noise gain): the same read with
        y + T * sigma_unit(m, y) * randn per lane, clipped, then rank-cut."""
        y, m = self._y_m(aat, per_group)
        return rank_cut(_lane.sample_read(y, m, T, self.noise, generator), self.Vt, self.N)

    # ----- learning -----

    def adapt(self, aat, target, feedback="hard", per_group=False):
        """One supervised step: adapting FF read on every lane, then per lane
        target -> RH, fired (y > 0) -> RL, else RF; recode the FF read it just saw.

        ``target`` is an output-space AAT ``[..., t]`` of lane indices (-1 = NONE padding);
        grouped modules take ``[..., G, t]`` (or ``[..., G]``), lane indices within each
        group. With ``per_group=True`` the input AAT is ``[..., G, k]`` and group g is
        taught on its own AAT. The whole leading batch is one step (stale reads within it);
        B = 1 is the bit-exact serial anchor. Voltage arithmetic mirrors the oracle
        expression for expression so the float64 rounding walk is identical.

        ``feedback`` selects the rule for a non-target lane that fired (y > 0). "hard" drives it
        down with RL, which maximizes the classification margin and is what a classifier wants.
        "soft" drops the RL term, so that lane decays (RF) with everyone else. With many lanes
        each example is one positive against L-1 negatives, and the RL punishment on co-firing
        lanes dominates what the weights converge to; drop it and the bank keeps every answer the
        data supports, which is what you want to sample from. "hard" is the original schedule,
        bit-exact.
        """
        feedback = check_feedback(feedback)
        if self.frozen:
            raise RuntimeError("frozen pack weights cannot adapt; lift live state via from_core")
        self._check_per_group(per_group)
        self._bag = {}                        # this step mutates ga/gb; drop the read tables
        K, L, G = self.num_spaces, self.num_lanes, self.num_groups
        gshape = () if G is None else (G,)
        k_axes = 2 if per_group else 1
        lead = aat.shape[:-k_axes]
        a = aat.reshape(-1, G, K) if per_group else aat.reshape(-1, K)
        B = a.shape[0]
        tgt = target.reshape(B, *gshape, -1).to(torch.int64)

        # FF read at full drive; the read itself drives every active synapse.
        top, bot = self._read_sums(a, per_group, fresh=True)
        dt = _lane.y_dtype(top.device)
        y = _lane.divide(top, bot, dt)                              # [B, *gshape, L]
        vy = self.vf * y
        dga = java_round(self.vf - vy).to(torch.int32)
        dgb = java_round(vy + self.vf).to(torch.int32)
        active_read = bot != 0                                      # all-NONE drives nothing
        _lane.apply_update(self.ga, self.gb, a, dga, dgb, lane_mask=active_read,
                           per_group=per_group)

        # Instruction select from the FF read. NONE target entries park in a spare column so
        # a -1 pad can never scatter-collide with a real target on lane 0.
        tmask = torch.zeros(B, *gshape, L + 1, dtype=torch.bool, device=a.device)
        tmask.scatter_(-1, torch.where(tgt >= 0, tgt, L), True)
        tmask = tmask[..., :L]
        fired = y > 0
        is_rl = (fired & ~tmask) if feedback == "hard" else torch.zeros_like(tmask)

        # RF is a read: its update uses the fresh post-FF y of its own lane.
        top2, bot2 = self._read_sums(a, per_group, fresh=True)
        y2 = _lane.divide(top2, bot2, dt)
        vy_rf = self.vr * y2
        dga_rf = java_round(self.vr - vy_rf).to(torch.int32)
        dgb_rf = java_round(vy_rf + self.vr).to(torch.int32)
        vy_rh = 1.0 * self.vr                                       # coeff +1, no H
        dga_rh = int(java_round(torch.tensor(self.vr - vy_rh)).item())
        dgb_rh = int(java_round(torch.tensor(vy_rh + self.vr)).item())
        vy_rl = -1.0 * self.vr
        dga_rl = int(java_round(torch.tensor(self.vr - vy_rl)).item())
        dgb_rl = int(java_round(torch.tensor(vy_rl + self.vr)).item())

        dga_fb = torch.where(tmask, dga_rh, torch.where(is_rl, dga_rl, dga_rf))
        dgb_fb = torch.where(tmask, dgb_rh, torch.where(is_rl, dgb_rl, dgb_rf))
        _lane.apply_update(self.ga, self.gb, a, dga_fb, dgb_fb, lane_mask=active_read,
                           per_group=per_group)

        width = L if self.N is None else self.N
        return rank_cut(y, self.Vt, self.N).reshape(*lead, *gshape, width)

    def extra_repr(self):
        tier = "frozen int8 pack" if self.frozen else "live int32"
        groups = "" if self.num_groups is None else f"groups={self.num_groups}, "
        return (f"{groups}lanes={self.num_lanes}, spaces={self.num_spaces}, "
                f"channels={self.num_channels}, Vt={self.Vt}, N={self.N}, {tier}")
