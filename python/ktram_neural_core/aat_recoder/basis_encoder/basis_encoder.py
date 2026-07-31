"""BasisEncoder — the second L1 AAT recoder: an unsupervised WTA codebook.

Where RankCut (the first L1 recoder) is supervised — read every lane, drive the label's lane up and
the confident-wrong lanes down — the basis encoder removes the label and points the same routine at
its own read winner. That one substitution (target = read winner, not target = label) turns the
supervised classifier into an unsupervised codebook learner. Everything downstream is identical: a
sub-threshold FFLV decide, then balanced FF + reverse (RH up / RL down) corrections on the winner
and on the other lanes that fired.

A plain winner-take-all group collapses — a few lanes win early, get rewarded, win more, and the
rest go dead, so the codebook shrinks to a handful of entries answering for everything. Two
stabilizers, working against each other, prevent that:

  exclusion    a lane already rewarded this cycle steps aside — no second reward this cycle. This
               spreads reward across lanes instead of piling it on an early leader. It is the
               anti-collapse mechanism: without it a few lanes win everything.
  recruitment  once `gather_abandon` updates have passed without the cycle completing, force-reward
               the highest-reading lane that has not yet won. This pulls idle lanes into service and
               keeps the cycle turning; measured on top of exclusion it sharpens the codebook.

A CYCLE is the bookkeeping that arms both: one bit per lane records who has won; when every lane has
won at least once, the cycle completes and the bits clear. Together the two drive every lane toward
roughly one reward per cycle, which is what keeps a competitive group using its whole width. In
hardware this is one bit per lane plus a counter.

The routine is decide-then-correct. The FFLV decide is sub-threshold: it costs nothing, disturbs no
conductance, and owes no reverse partner. Only then are forward instructions spent, and only on the
winner and the few lanes that fired — on a wide group most lanes get no instruction on a given
update. Every FF is issued inside `_correct` alongside its reverse partner, so balance is structural
rather than something to assert.

Ported from Knowm's basis-feature work (the routine that learns basis features on MNIST without
labels). The public emulator is the reference.
"""

from ..base import AATRecoder


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _crossbar_geometry(max_channels):
    """One 1 x next_pow2 crossbar shape per space, sized to the widest input space (matches
    classify.linear and the RankCut transform). Narrower spaces use low addresses in it."""
    return 1, _next_pow2(max(1, max_channels))


class GatherAbandon:
    """Cycle bookkeeping for one WTA group: one bit per lane plus a counter.

    The buffer holds lanes that have WON a read this cycle. When every lane has won, the cycle
    completes and the buffer clears. `count` is adapt calls since the last completion; once it
    reaches `gather_abandon`, recruitment is armed.
    """

    def __init__(self, num_channels, gather_abandon):
        self.num_channels = num_channels
        self.gather_abandon = gather_abandon
        self._buffer = set()
        self._count = 0
        self.cycles = 0            # completed cycles (buffer cleared)
        self.cycle_lengths = []    # adapt calls per completed cycle

    def has_won(self, lane):
        return lane in self._buffer

    def mark(self, lane):
        if lane is not None:
            self._buffer.add(lane)

    def should_recruit(self):
        return self.gather_abandon is not None and self._count >= self.gather_abandon

    def recruit(self, y):
        """The highest-reading lane that has not won this cycle, or None if all have."""
        for lane in sorted(range(len(y)), key=lambda i: y[i], reverse=True):
            if lane not in self._buffer:
                return lane
        return None

    def tick(self):
        """One adapt call has completed. Clear the cycle if every lane has been claimed."""
        self._count += 1
        if len(self._buffer) >= self.num_channels:
            self.cycle_lengths.append(self._count)
            self.cycles += 1
            self._buffer.clear()
            self._count = 0

    def reset_cycle(self):
        """Abandon an incomplete cycle: clear the won-buffer and counter without every lane having
        won. This releases exclusion's throttle so learning keeps going, instead of freezing when a
        few lanes stop winning and the buffer never fills."""
        self.cycle_lengths.append(self._count)
        self.cycles += 1
        self._buffer.clear()
        self._count = 0

    @property
    def count(self):
        return self._count


class BasisGroup(AATRecoder):
    """One WTA group of `channels` neural lanes: AAT in, one winner out — the code for the input.

    read(in_aat)   -> winner            non-disturbing FFLV read, argmax (no feedback)
    adapt(in_aat)  -> winner            one unsupervised update (decide, reward, depress, recruit)

    exclusion and recruitment default on; turn either off to reproduce the collapse (exclusion off)
    or the dead-lane (recruitment off) failure modes. `gather_abandon` sets the stall cadence; it
    defaults to one nominal cycle (`channels`).

    `abandon_action` sets what happens when a cycle stalls (see `adapt`): "recruit" (with recruitment
    on) force-rewards an idle lane to keep the whole width live; "reset" (recruitment off) clears the
    cycle without rewarding anyone, so the lanes that win keep sharpening and non-competitive lanes
    fade to init — a self-pruning codebook where dead lanes end at win count 0. Both stall responses
    are hardware-native: exclusion is one bit per lane, reset clears those bits, recruit reads and
    rewards. No post-hoc pruning or off-chip crispness metric is involved.
    """

    def __init__(self, core, channels, *, gather_abandon=None, exclusion=True, recruitment=True,
                 abandon_action="recruit"):
        super().__init__(core, list(range(channels)))
        if abandon_action not in ("recruit", "reset"):
            raise KeyError(f"unknown abandon_action {abandon_action!r}; valid: 'recruit', 'reset'")
        self.channels = channels
        self.exclusion = exclusion
        self.recruitment = recruitment
        self.abandon_action = abandon_action
        if gather_abandon is None and (recruitment or abandon_action == "reset"):
            gather_abandon = channels
        self._ga = GatherAbandon(channels, gather_abandon)
        # instrumentation for the visuals
        self.n_adapt = 0
        self.n_throttled = 0     # winner had already won this cycle, so no feedback was issued
        self.n_recruited = 0     # forced reward to an idle lane
        self.n_instructions = 0  # kT-RAM instructions spent on feedback
        self.win_counts = [0] * channels

    # -- configuration ------------------------------------------------------

    @property
    def gather_abandon(self):
        return self._ga.gather_abandon

    @gather_abandon.setter
    def gather_abandon(self, value):
        self._ga.gather_abandon = value

    # -- reading (non-disturbing) -------------------------------------------

    def read_scores(self, in_aat):
        """The sub-threshold y vector, one per lane. Non-disturbing, so free to issue."""
        return [self.core.evaluate(in_aat, "FFLV", lane) for lane in self._lanes]

    def winner(self, in_aat):
        y = self.read_scores(in_aat)
        return max(self._lanes, key=lambda i: y[i])

    def read(self, in_aat):
        """The code for `in_aat`: the single lane that reads highest. Leaves no mark."""
        return self.winner(in_aat)

    # -- learning -----------------------------------------------------------

    def adapt(self, in_aat):
        """One unsupervised update. The read winner is its own target."""
        # 1. decide — costs nothing, disturbs nothing, owes nothing
        y = self.read_scores(in_aat)
        winner = max(self._lanes, key=lambda i: y[i])

        # 2. exclusion: a lane already rewarded this cycle issues no feedback at all
        throttled = self.exclusion and self._ga.has_won(winner)
        if throttled:
            self.n_throttled += 1
        else:
            # 3. reward the winner, depress the other lanes that fired
            self._correct(in_aat, winner, "RH")
            for lane in self._lanes:
                if lane != winner and y[lane] > 0:
                    self._correct(in_aat, lane, "RL")

        # 4. gather-abandon safety: after gather_abandon idle updates the cycle has stalled (some
        #    lanes never win, so the won-buffer never fills). Three responses (the abandon_action
        #    choice is decoupled from the recruitment flag, so each is independently reachable):
        #      recruitment on,  "recruit" — force-reward the highest-reading unclaimed lane (the Java
        #                                   behavior): props up the whole width, every lane stays live.
        #      recruitment off, "reset"   — clear the incomplete cycle without rewarding anyone, so
        #                                   exclusion's throttle releases and the lanes that actually
        #                                   win keep winning and sharpen, while non-competitive lanes
        #                                   are never rewarded and fade back toward init. The codebook
        #                                   self-prunes: dead lanes go dark (win count -> 0), no
        #                                   post-hoc pruning step or crispness metric needed.
        #      recruitment off, no reset  — do nothing: the cycle never clears and exclusion freezes
        #                                   further reward (the 'recruitment off' ablation).
        if self._ga.should_recruit():
            if self.recruitment and self.abandon_action == "recruit":
                r = self._ga.recruit(y)
                if r is not None:
                    self._correct(in_aat, r, "RH")
                    self._ga.mark(r)
                    self.n_recruited += 1
            elif self.abandon_action == "reset":
                self._ga.reset_cycle()

        # 5. every other lane gets no instruction at all

        # 6. bookkeeping — the cycle turns whether or not anything was spent
        self._ga.mark(winner)
        self._ga.tick()
        self.n_adapt += 1
        self.win_counts[winner] += 1
        return winner

    def _correct(self, in_aat, lane, reverse_instr):
        """The only place a forward instruction is issued. No path leaves an FF unpaired."""
        self.core.evaluate(in_aat, "FF", lane)
        self.core.evaluate(in_aat, reverse_instr, lane)
        self.n_instructions += 2

    # -- instrumentation ----------------------------------------------------

    @property
    def throttle_rate(self):
        return self.n_throttled / self.n_adapt if self.n_adapt else 0.0

    @property
    def recruit_rate(self):
        return self.n_recruited / self.n_adapt if self.n_adapt else 0.0

    @property
    def mean_cycle_length(self):
        lens = self._ga.cycle_lengths
        return sum(lens) / len(lens) if lens else float("nan")

    @property
    def codebook_utilization(self):
        """Fraction of lanes that have won at least once since the counters were reset."""
        return sum(1 for c in self.win_counts if c) / self.channels

    @property
    def winner_entropy(self):
        """Entropy of the winner histogram in bits — catches a codebook that uses every lane but
        leans hard on three."""
        import math
        total = sum(self.win_counts)
        if not total:
            return 0.0
        h = 0.0
        for c in self.win_counts:
            if c:
                p = c / total
                h -= p * math.log2(p)
        return h

    def reset_stats(self):
        """Clear the per-epoch counters. Cycle state belongs to training and is left alone."""
        self.n_adapt = 0
        self.n_throttled = 0
        self.n_recruited = 0
        self.n_instructions = 0
        self.win_counts = [0] * self.channels
        self._ga.cycle_lengths = []


class BasisEncoder:
    """A bank of `n_groups` BasisGroups over one shared input AAT.

    Each group is an independent WTA codebook of `channels` lanes; the encoder's output AAT is the
    tuple of group winners, one channel per group. One group already learns a codebook; a bank gives
    a wider, more separable code (each group is seeded differently, so they specialize differently).

        read(in_aat)   -> out_aat   tuple of group winners (non-disturbing)
        adapt(in_aat)  -> out_aat   one unsupervised update per group

    Sharp reads by default (read_noise = 0); the byte model puts a quantized device at every synapse.
    """

    def __init__(self, space_sizes, n_groups=1, channels=32, *, model="byte", init="low",
                 gather_abandon=None, exclusion=True, recruitment=True, read_noise=0, seed=None,
                 **core_kwargs):
        from ...core import Core
        self.space_sizes = list(space_sizes)
        self.n_groups = n_groups
        self.channels = channels
        rows, cols = _crossbar_geometry(max(self.space_sizes))
        self.groups = []
        for g in range(n_groups):
            core = Core(
                rows,
                cols,
                spaces_per_lane=len(self.space_sizes),
                num_lanes=channels,
                model=model,
                init=init,
                read_noise=read_noise,
                seed=(None if seed is None else seed + g),
                **core_kwargs,
            )
            self.groups.append(
                BasisGroup(core, channels, gather_abandon=gather_abandon,
                           exclusion=exclusion, recruitment=recruitment)
            )

    @property
    def out_space_sizes(self):
        """The output AAT's spaces: one per group, each of width `channels`."""
        return [self.channels] * self.n_groups

    def read(self, in_aat):
        return tuple(group.read(in_aat) for group in self.groups)

    def adapt(self, in_aat):
        return tuple(group.adapt(in_aat) for group in self.groups)

    def set_gather_abandon(self, value):
        for group in self.groups:
            group.gather_abandon = value

    def reset_stats(self):
        for group in self.groups:
            group.reset_stats()
