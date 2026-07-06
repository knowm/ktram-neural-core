# RankCut — L1 AAT Recoder

The first L1 AAT recoder: a supervised classifier as one object. One neural lane per label,
wrapped behind an AAT-level interface with all the analog contained inside.

- `rank_cut.py` — the Python emulator: the `RankCut` recoder (`read` / `adapt`) and its pure
  `rank_cut(y, Vt, N)` readout policy.

The readout policy keeps lanes with `Vy ≥ Vt` and returns them rank-ordered (strongest first),
capped at `N`. `(Vt, N)` recovers winner, winner-above-zero, top-k, and all-above-zero from one
block. Teaching is pinned to the lane's own sign (`Vy > 0`), independent of `(Vt, N)`.
