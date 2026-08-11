# BasisEncoder — L1 AAT Recoder (unsupervised)

The second L1 AAT recoder: an unsupervised winner-take-all codebook as one object. A group of
neural lanes competes over the input AAT; over many inputs the lanes specialize, each answering for
one recurring pattern. That set of specialized lanes is the basis (equivalently the codebook, or
dictionary).

- `basis_encoder.py` — the emulator: `BasisGroup` (one WTA group, `read` / `learn`), `BasisEncoder`
  (a bank of groups over one shared input, `read` / `learn`), and `GatherAbandon` (the cycle
  bookkeeping).

## One routine, label optional

This is the same lane-driving routine as [RankCut](../rank_cut/), the supervised L1 recoder, with a
single substitution — the answer to "which lane is the target":

- **RankCut (supervised):** target = the label's lane.
- **BasisEncoder (unsupervised):** target = the read winner.

Everything downstream is identical: a sub-threshold `FFLV` decide, then balanced `FF` + reverse
(`RH` up on the target, `RL` down on the other lanes that fired). One hardware module serves both
the labeled classifier and the unlabeled codebook.

## The two stabilizers

A plain winner-take-all group collapses onto a few lanes. Two mechanisms, working against each
other, keep it using its whole width:

- **exclusion** — a lane already rewarded this cycle steps aside, so reward spreads instead of
  piling on an early leader. This is the anti-collapse mechanism: turn it off and a few lanes win
  everything (utilization, coverage, and purity all fall together).
- **recruitment** — after `gather_abandon` idle updates, force-reward the highest-reading unclaimed
  lane, so no lane stays dead and the cycle keeps turning. Its measured effect is a consolidation /
  purity refinement on top of exclusion, strongest when there are many patterns.

A **cycle** is the bookkeeping that arms both: one bit per lane records who has won; when every lane
has won, the cycle clears. In hardware, one bit per lane plus a counter. Both default on; turn
either off (`exclusion=False`, `recruitment=False`) to see its contribution.

## Self-pruning (form, then withdraw recruitment)

`recruitment=True` keeps every lane live, which is what you want while the basis is still forming.
Once it has formed, withdrawing recruitment turns the group into a self-pruning codebook. Set
`recruitment=False, abandon_action="reset"`: on a stalled cycle, instead of force-rewarding an idle
lane the group just **resets** the cycle (clears the won-bits), so exclusion keeps the lanes that
still win cycling and sharpening. Nothing props up a lane that stops winning, so non-competitive
lanes are depressed back toward init and fade. The codebook prunes itself down to what the data
supports, and the survivors are read out by **win count alone** — one counter per lane, no off-chip
metric. Run it as two phases (recruitment on to form, then off to sharpen); the examples do exactly
this.

The hardware realization is in [`hardware.md`](hardware.md).
