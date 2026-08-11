"""BasisEncoder — the L1 unsupervised WTA codebook, group-stacked, in torch.

The oracle's BasisGroup routine with the per-lane Python loops collapsed into tensor ops:
one batched sub-threshold read scores every lane of every group, the winner/exclusion/
recruitment logic resolves over that small score matrix, and one scatter-style update per
phase applies the corrections. The group axis is free batching — the generator's 16 patch
codebooks are one module with ``[16, 64, 49, 2]`` weights, exactly the shape the export
already writes.

Verbs, mirroring the oracle:

    read(aat)   -> out_aat   one winner symbol per group, ``[..., G]`` (non-disturbing)
    adapt(aat)  -> out_aat   one unsupervised update: decide, reward, depress, recruit

Input AATs are ``[..., k]`` int64 (-1 = NONE), shared by every group; with ``per_group=True``
they are ``[..., G, k]`` and group g reads its own AAT (the generator's per-patch feed).

The GatherAbandon cycle state is exactly the oracle's — one won-bit per lane plus a counter
per group — held as registered buffers so ``state_dict()`` carries the live training state.
(`cycle_lengths`, a ragged per-group history, stays a plain Python list and is not
serialized.) The three stall responses (recruit / reset / freeze) and the mid-training
config flips the generator performs (``recruitment = False``, ``abandon_action = "reset"``)
work the same way here.

Batching semantics (spec 08 §6): B = 1 is the bit-exact serial anchor — corrections read
fresh state in oracle order, including the recruit re-read after the winner/fired
corrections have landed. B > 1 is the stale-read batch: all decisions and correction
voltages come from the batch's one score read, bookkeeping resolves example by example in
order, and each phase's deltas accumulate before one clamp. Qualified by the tier-3
quality-vs-B bands, not apologized for. Byte model only.
"""

import numpy as np
import torch
from torch import nn

from . import _lane
from ._lane import NoiseParams, java_round, y_dtype
from .classifier import FORWARD_LOW_V, FORWARD_V, REVERSE_V


class BasisEncoder(nn.Module):
    """A bank of ``n_groups`` WTA groups of ``channels`` lanes over ``num_spaces`` spaces of
    ``num_channels`` channels each. Weights are int32 buffers in [1, 255] (live) or the int8
    frozen pack tier (``from_pack``; reads only).

    ``seed`` may be an int (group g draws from seed + g, the oracle BasisEncoder rule), a
    list of per-group seeds (the generator's explicitly-seeded groups), or None.
    """

    def __init__(self, n_groups, channels, num_spaces, num_channels, *,
                 gather_abandon=None, exclusion=True, recruitment=True,
                 abandon_action="recruit", init="low", seed=None,
                 forward_voltage=FORWARD_V, reverse_voltage=REVERSE_V,
                 forward_low_voltage=FORWARD_LOW_V, noise=None):
        super().__init__()
        if abandon_action not in ("recruit", "reset"):
            raise KeyError(f"unknown abandon_action {abandon_action!r}; valid: 'recruit', 'reset'")
        self.n_groups = n_groups
        self.channels = channels
        self.num_spaces = num_spaces
        self.num_channels = num_channels
        self.exclusion = exclusion
        self.recruitment = recruitment
        self.abandon_action = abandon_action
        if gather_abandon is None and (recruitment or abandon_action == "reset"):
            gather_abandon = channels
        self.gather_abandon = gather_abandon
        self.vf = float(forward_voltage)
        self.vr = float(reverse_voltage)
        self.flv = float(forward_low_voltage)
        if java_round(torch.tensor(2.0 * abs(self.flv))).item() != 0.0:
            raise ValueError(
                f"forward_low_voltage {self.flv} would disturb a byte read; keep |flv| < 0.25")
        self.noise = noise if noise is not None else NoiseParams(v_read=abs(self.flv))
        self.frozen = False
        # CPU reads run as an embedding bag over a derived table (the 08d race winner),
        # cached per read mode (per_group needs a different row layout than the shared
        # read) and dropped on any adapt. Hand-mutating ga/gb requires clearing it.
        self._bag = {}

        seeds = self._seeds(seed)
        ga_parts, gb_parts = [], []
        for g in range(n_groups):
            ga_g, gb_g = _lane.init_weights((channels, num_spaces, num_channels), init, seeds[g])
            ga_parts.append(ga_g)
            gb_parts.append(gb_g)
        self.register_buffer("ga", torch.stack(ga_parts))
        self.register_buffer("gb", torch.stack(gb_parts))

        # GatherAbandon cycle state + instrumentation, all per group.
        self.register_buffer("won", torch.zeros(n_groups, channels, dtype=torch.bool))
        self.register_buffer("count", torch.zeros(n_groups, dtype=torch.int64))
        self.register_buffer("cycles", torch.zeros(n_groups, dtype=torch.int64))
        self.register_buffer("win_counts", torch.zeros(n_groups, channels, dtype=torch.int64))
        self.register_buffer("n_adapt", torch.zeros(n_groups, dtype=torch.int64))
        self.register_buffer("n_throttled", torch.zeros(n_groups, dtype=torch.int64))
        self.register_buffer("n_recruited", torch.zeros(n_groups, dtype=torch.int64))
        self.register_buffer("n_instructions", torch.zeros(n_groups, dtype=torch.int64))
        self.cycle_lengths = [[] for _ in range(n_groups)]

    def _seeds(self, seed):
        if seed is None:
            return [None] * self.n_groups
        if isinstance(seed, (list, tuple)):
            if len(seed) != self.n_groups:
                raise ValueError(f"{len(seed)} seeds for {self.n_groups} groups")
            return list(seed)
        return [seed + g for g in range(self.n_groups)]

    # ----- lifting trained oracle state -----

    @classmethod
    def from_core(cls, source):
        """Lift a trained oracle codebook — a BasisEncoder, a BasisGroup, or a list of
        BasisGroups (the generator's per-patch groups) — into one stacked torch module,
        carrying the full training state: weights, cycle bits, counters, win statistics."""
        from ..aat_recoder import BasisEncoder as OracleEncoder
        from ..aat_recoder import BasisGroup

        if isinstance(source, OracleEncoder):
            groups = source.groups
        elif isinstance(source, BasisGroup):
            groups = [source]
        else:
            groups = list(source)
        g0 = groups[0]
        core0 = g0.core
        if core0.model_name != "byte":
            raise NotImplementedError(
                f"torch backend implements the byte model only, got {core0.model_name!r}")
        G = len(groups)
        L = g0.channels
        K = core0.spaces_per_lane
        S = core0.crossbar_rows * core0.crossbar_cols
        self = cls(G, L, K, S, gather_abandon=g0.gather_abandon, exclusion=g0.exclusion,
                   recruitment=g0.recruitment, abandon_action=g0.abandon_action,
                   init="low_noiseless", forward_voltage=core0.forward_voltage,
                   reverse_voltage=core0.reverse_voltage,
                   forward_low_voltage=core0.forward_low_voltage,
                   noise=NoiseParams.from_core(core0))
        ga = np.empty((G, L, K, S), dtype=np.int32)
        gb = np.empty((G, L, K, S), dtype=np.int32)
        for g, grp in enumerate(groups):
            for lane_i in range(L):
                lane = grp.core.lane(lane_i)
                for k, space in enumerate(lane.spaces):
                    ga[g, lane_i, k] = [int(space.a.device_at(s).g()) for s in range(S)]
                    gb[g, lane_i, k] = [int(space.b.device_at(s).g()) for s in range(S)]
            for lane_i in range(L):
                self.won[g, lane_i] = grp._ga.has_won(lane_i)
            self.count[g] = grp._ga.count
            self.cycles[g] = grp._ga.cycles
            self.cycle_lengths[g] = list(grp._ga.cycle_lengths)
            self.win_counts[g] = torch.tensor(grp.win_counts)
            self.n_adapt[g] = grp.n_adapt
            self.n_throttled[g] = grp.n_throttled
            self.n_recruited[g] = grp.n_recruited
            self.n_instructions[g] = grp.n_instructions
        self.ga.copy_(torch.from_numpy(ga))
        self.gb.copy_(torch.from_numpy(gb))
        return self

    # ----- the frozen pack tier -----

    @classmethod
    def from_pack(cls, path):
        """Load the encoder section of a frozen generator-weights export. Reads only;
        ``adapt`` raises."""
        from .pack import load_section
        qdiff, qmag, diff_scale, mag_scale, noise = load_section(path, "encoder")
        G, L, K, S = qdiff.shape
        self = cls(G, L, K, S, init="low_noiseless",
                   noise=NoiseParams(**{k: v for k, v in noise.items() if k != "v_fflv"},
                                     v_read=noise["v_fflv"]) if noise else None)
        self.frozen = True
        delattr(self, "ga")
        delattr(self, "gb")
        self.register_buffer("qdiff", torch.from_numpy(qdiff))
        self.register_buffer("qmag", torch.from_numpy(qmag))
        self.diff_scale = float(diff_scale)
        self.mag_scale = float(mag_scale)
        return self

    def to_pack(self, path):
        """Quantize and write this module's weights as the encoder section of a
        generator-weights ``.npz`` pack (creating or updating the file in place)."""
        from .pack import save_section
        save_section(path, "encoder", *self._diff_mag_float(), self.noise)

    def _diff_mag_float(self):
        if self.frozen:
            return (self.qdiff.numpy().astype(np.float32) * self.diff_scale,
                    self.qmag.numpy().astype(np.float32) * self.mag_scale)
        ga = self.ga.cpu().numpy().astype(np.float32)
        gb = self.gb.cpu().numpy().astype(np.float32)
        return ga - gb, ga + gb

    # ----- reading (non-disturbing) -----

    def _load_from_state_dict(self, *args, **kwargs):
        self._bag = {}                        # loaded weights invalidate the read tables
        super()._load_from_state_dict(*args, **kwargs)

    def _read_sums(self, aat, per_group, fresh):
        """The (top, bottom) sums: embedding-bag table on CPU reads, the direct gather
        elsewhere and for the fresh reads inside adapt (the table would be stale)."""
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
            return _lane.bag_read_sums(
                self._bag[key], aat, self.num_spaces, self.num_channels,
                G=self.n_groups if per_group else None,
                out_shape=(self.n_groups, self.channels))
        return _lane.read_sums(w1, w2, aat, per_group=per_group,
                               paired=not self.frozen)

    def _y_m(self, aat, per_group=False, fresh=False):
        top, bot = self._read_sums(aat, per_group, fresh)
        dt = y_dtype(top.device)
        if self.frozen:
            y = _lane.divide(top, bot, dt, self.diff_scale, self.mag_scale)
            return y, bot.to(dt) * self.mag_scale
        return _lane.divide(top, bot, dt), bot.to(dt)

    def read_scores(self, aat, per_group=False):
        """The sub-threshold y matrix, ``[..., G, L]`` — debug/instrumentation."""
        return self._y_m(aat, per_group)[0]

    def read(self, aat, per_group=False):
        """The code for ``aat``: the winner lane per group, ``[..., G]``. Leaves no mark."""
        return self.read_scores(aat, per_group).argmax(dim=-1)

    def read_sampled(self, aat, T, generator=None, per_group=False):
        """Opt-in noisy read at temperature T: each lane's y carries T * sigma_unit(m, y)
        noise, so the winner is a sample, not the argmax — the generator's temperature knob."""
        y, m = self._y_m(aat, per_group)
        return _lane.sample_read(y, m, T, self.noise, generator).argmax(dim=-1)

    # ----- learning -----

    def adapt(self, aat, per_group=False):
        """One unsupervised update per group; the read winner is its own target.

        Decide (one batched sub-threshold read), exclusion throttle, reward the winner
        (FF + RH) and depress the other fired lanes (FF + RL), then the gather-abandon stall
        response, then the cycle bookkeeping — the oracle's order exactly.
        """
        if self.frozen:
            raise RuntimeError("frozen pack weights cannot adapt; lift live state via from_core")
        self._bag = {}                        # this step mutates ga/gb; drop the read tables
        G, L = self.n_groups, self.channels
        k_axes = 2 if per_group else 1
        lead = aat.shape[:-k_axes]
        a = aat.reshape(-1, G, self.num_spaces) if per_group else \
            aat.reshape(-1, self.num_spaces)
        B = a.shape[0]

        y, _ = self._y_m(a, per_group, fresh=True)              # [B, G, L], stale for the batch
        winners = y.argmax(dim=-1)                              # [B, G]

        if B == 1:
            self._adapt_serial(a, y[0], winners[0], per_group)
        else:
            self._adapt_batch(a, y, winners, per_group)
        return winners.reshape(*lead, G)

    # The RH/RL feedback deltas are y-independent; mirror the oracle's expressions.
    def _feedback_consts(self):
        vy_rh = 1.0 * self.vr
        vy_rl = -1.0 * self.vr
        jr = lambda v: int(java_round(torch.tensor(v)).item())  # noqa: E731
        return (jr(self.vr - vy_rh), jr(vy_rh + self.vr),       # RH: (dga, dgb)
                jr(self.vr - vy_rl), jr(vy_rl + self.vr))       # RL: (dga, dgb)

    def _ff_deltas(self, y):
        vy = self.vf * y
        return (java_round(self.vf - vy).to(torch.int32),
                java_round(vy + self.vf).to(torch.int32))

    def _adapt_serial(self, a, y, win, per_group):
        """B = 1: the oracle's serial sequence, bit for bit, vectorized across groups."""
        G, L = self.n_groups, self.channels
        gi = torch.arange(G, device=y.device)
        rh_a, rh_b, rl_a, rl_b = self._feedback_consts()

        throttled = self.won[gi, win] if self.exclusion else torch.zeros(G, dtype=torch.bool)
        reward = torch.zeros(G, L, dtype=torch.bool, device=y.device)
        reward[gi, win] = ~throttled
        punish = ~throttled[:, None] & (y > 0)
        punish[gi, win] = False
        act = reward | punish
        if act.any():
            dga_ff, dgb_ff = self._ff_deltas(y)
            _lane.apply_update(self.ga, self.gb, a, dga_ff, dgb_ff,
                               lane_mask=act, per_group=per_group)
            dga_fb = reward * rh_a + punish * rl_a
            dgb_fb = reward * rh_b + punish * rl_b
            _lane.apply_update(self.ga, self.gb, a, dga_fb.to(torch.int32),
                               dgb_fb.to(torch.int32), lane_mask=act, per_group=per_group)
        self.n_throttled += throttled
        self.n_instructions += 2 * act.sum(dim=-1)

        # Gather-abandon stall response (checked every call, oracle order: after corrections,
        # before the winner is marked).
        stalled = self._stalled()
        if stalled.any():
            if self.recruitment and self.abandon_action == "recruit":
                masked = y.masked_fill(self.won, float("-inf"))
                r = masked.argmax(dim=-1)                       # highest unclaimed, first on ties
                do = stalled & ~self.won.all(dim=-1)
                if do.any():
                    # The recruit correction is issued last, so its FF reads the state the
                    # winner/fired corrections just left behind — a fresh read, not the stale one.
                    y2, _ = self._y_m(a, per_group, fresh=True)
                    y2 = y2.reshape(G, L)
                    rmask = torch.zeros(G, L, dtype=torch.bool, device=y.device)
                    rmask[gi, r] = do
                    dga_ff, dgb_ff = self._ff_deltas(y2)
                    _lane.apply_update(self.ga, self.gb, a, dga_ff, dgb_ff,
                                       lane_mask=rmask, per_group=per_group)
                    _lane.apply_update(self.ga, self.gb, a,
                                       (rmask * rh_a).to(torch.int32),
                                       (rmask * rh_b).to(torch.int32),
                                       lane_mask=rmask, per_group=per_group)
                    self.won[gi, r] |= do
                    self.n_recruited += do
                    self.n_instructions += 2 * do
            elif self.abandon_action == "reset":
                self._reset_cycles(stalled)

        self._mark_and_tick(win)

    def _adapt_batch(self, a, y, winners, per_group):
        """B > 1: one stale-read step. Decisions and correction voltages all come from the
        batch's single score read; bookkeeping resolves example by example in order; each
        phase's deltas accumulate, then one clamp."""
        G, L = self.n_groups, self.channels
        B = y.shape[0]
        rh_a, rh_b, rl_a, rl_b = self._feedback_consts()
        y_np = y.cpu().numpy()
        win_np = winners.cpu().numpy()
        won = self.won.cpu().numpy().copy()
        count = self.count.cpu().numpy().copy()
        cycles = self.cycles.cpu().numpy().copy()
        gi = np.arange(G)

        ff_count = np.zeros((B, G, L), dtype=np.int32)          # a lane can be corrected twice
        rh = np.zeros((B, G, L), dtype=bool)
        rl = np.zeros((B, G, L), dtype=bool)
        n_thr = np.zeros(G, dtype=np.int64)
        n_rec = np.zeros(G, dtype=np.int64)
        wc = np.zeros((G, L), dtype=np.int64)    # win_counts increments, applied once at
                                                 # the end (indexing a buffer with numpy
                                                 # arrays breaks on non-CPU devices)

        for b in range(B):
            win = win_np[b]
            throttled = won[gi, win] if self.exclusion else np.zeros(G, dtype=bool)
            reward = np.zeros((G, L), dtype=bool)
            reward[gi, win] = ~throttled
            punish = ~throttled[:, None] & (y_np[b] > 0)
            punish[gi, win] = False
            rh[b] |= reward
            rl[b] |= punish
            ff_count[b] += reward + punish
            n_thr += throttled

            stalled = (self.gather_abandon is not None) & (count >= (self.gather_abandon or 0))
            if stalled.any():
                if self.recruitment and self.abandon_action == "recruit":
                    masked = np.where(won, -np.inf, y_np[b])
                    r = masked.argmax(axis=-1)
                    do = stalled & ~won.all(axis=-1)
                    rh[b, gi[do], r[do]] = True
                    ff_count[b, gi[do], r[do]] += 1
                    won[gi[do], r[do]] = True
                    n_rec += do
                elif self.abandon_action == "reset":
                    for g in gi[stalled]:
                        self.cycle_lengths[g].append(int(count[g]))
                    cycles += stalled
                    won[stalled] = False
                    count[stalled] = 0

            won[gi, win] = True
            count += 1
            full = won.sum(axis=-1) >= L
            if full.any():
                for g in gi[full]:
                    self.cycle_lengths[g].append(int(count[g]))
                cycles += full
                won[full] = False
                count[full] = 0
            wc[gi, win_np[b]] += 1

        dga_ff, dgb_ff = self._ff_deltas(y)
        ffc = torch.from_numpy(ff_count).to(y.device)
        rh_t = torch.from_numpy(rh).to(y.device)
        rl_t = torch.from_numpy(rl).to(y.device)
        _lane.apply_update(self.ga, self.gb, a, dga_ff * ffc, dgb_ff * ffc,
                           per_group=per_group)
        _lane.apply_update(self.ga, self.gb, a,
                           (rh_t * rh_a + rl_t * rl_a).to(torch.int32),
                           (rh_t * rh_b + rl_t * rl_b).to(torch.int32),
                           per_group=per_group)

        dev = self.won.device
        self.won.copy_(torch.from_numpy(won))
        self.count.copy_(torch.from_numpy(count))
        self.cycles.copy_(torch.from_numpy(cycles))
        self.win_counts += torch.from_numpy(wc).to(dev)
        self.n_throttled += torch.from_numpy(n_thr).to(dev)
        self.n_recruited += torch.from_numpy(n_rec).to(dev)
        self.n_instructions += 2 * (ffc.sum(dim=(0, 2)))
        self.n_adapt += B

    def _stalled(self):
        if self.gather_abandon is None:
            return torch.zeros(self.n_groups, dtype=torch.bool)
        return self.count >= self.gather_abandon

    def _reset_cycles(self, mask):
        """Abandon incomplete cycles: record the stalled count, clear the won-bits and
        counter without rewarding anyone (the self-pruning stall response)."""
        for g in torch.nonzero(mask).flatten().tolist():
            self.cycle_lengths[g].append(int(self.count[g]))
        self.cycles += mask
        self.won[mask] = False
        self.count[mask] = 0

    def _mark_and_tick(self, win):
        gi = torch.arange(self.n_groups, device=win.device)
        self.won[gi, win] = True
        self.count += 1
        full = self.won.sum(dim=-1) >= self.channels
        if full.any():
            for g in torch.nonzero(full).flatten().tolist():
                self.cycle_lengths[g].append(int(self.count[g]))
            self.cycles += full
            self.won[full] = False
            self.count[full] = 0
        self.win_counts[gi, win] += 1
        self.n_adapt += 1

    # ----- instrumentation (per group, mirroring the oracle's properties) -----

    @property
    def throttle_rate(self):
        return torch.where(self.n_adapt > 0, self.n_throttled / self.n_adapt.clamp(min=1),
                           torch.zeros_like(self.n_adapt, dtype=torch.float64))

    @property
    def recruit_rate(self):
        return torch.where(self.n_adapt > 0, self.n_recruited / self.n_adapt.clamp(min=1),
                           torch.zeros_like(self.n_adapt, dtype=torch.float64))

    @property
    def mean_cycle_length(self):
        return torch.tensor([sum(ls) / len(ls) if ls else float("nan")
                             for ls in self.cycle_lengths])

    @property
    def codebook_utilization(self):
        """Fraction of lanes per group that have won at least once since the last reset."""
        return (self.win_counts > 0).to(torch.float64).mean(dim=-1)

    @property
    def winner_entropy(self):
        """Entropy of each group's winner histogram in bits."""
        total = self.win_counts.sum(dim=-1, keepdim=True).clamp(min=1)
        p = self.win_counts.to(torch.float64) / total
        h = torch.where(p > 0, -p * torch.log2(p.clamp(min=1e-300)), torch.zeros_like(p))
        return h.sum(dim=-1)

    def reset_stats(self):
        """Clear the per-epoch counters. Cycle state belongs to training and is left alone."""
        self.n_adapt.zero_()
        self.n_throttled.zero_()
        self.n_recruited.zero_()
        self.n_instructions.zero_()
        self.win_counts.zero_()
        self.cycle_lengths = [[] for _ in range(self.n_groups)]

    def extra_repr(self):
        tier = "frozen int8 pack" if self.frozen else "live int32"
        return (f"groups={self.n_groups}, channels={self.channels}, "
                f"spaces={self.num_spaces}x{self.num_channels}, "
                f"gather_abandon={self.gather_abandon}, exclusion={self.exclusion}, "
                f"recruitment={self.recruitment}, abandon_action={self.abandon_action!r}, {tier}")
