# BasisEncoder — hardware realization

The basis encoder is an L1 AAT recoder: it drives a bank of neural lanes with kT-RAM instructions
and hides the analog behind a two-call interface (`read`, `learn`). This file describes how the
routine maps onto hardware. It reuses the readout of the [RankCut recoder](../rank_cut/hardware.md)
— the two L1 recoders share the same lane bank and the same swept-reference read — and adds only the
per-lane cycle bit that the unsupervised rule needs.

The Python in `basis_encoder.py` is the contract. The hardware below produces the same emitted
winner and the same feedback pattern; it does not model sweep dynamics or device noise.

## The signal

Each lane is a [2-1 memristor divider](../../../../../docs/architecture.md) and outputs one voltage
`Vy` in `[−V, +V]`, with zero the decision point (`Vy > 0` means the lane fired). A group is `S`
lanes over one shared input AAT. `read` and `learn` both start from the same read of all `S` lanes.

## The routine, decide then correct

One `learn` is four steps. Only the first is always paid; the rest touch a handful of lanes.

1. **Decide (sub-threshold, free).** Read every lane at the low read voltage (`FFLV`). This read is
   below the device threshold, so it disturbs no conductance and owes no reverse partner. It yields
   the per-lane `Vy`, and from them the winner (the argmax) and which lanes fired (`Vy > 0`). The
   winner is picked by the same swept-reference comparator readout RankCut uses — the ramp trips the
   strongest lane first, so the winner falls out of the timing with no per-lane ADC. `read` is this
   step alone.
2. **Reward the winner.** Drive the winning lane up: `FF` then `RH`. A balanced forward read plus its
   reverse partner.
3. **Depress the other fired lanes.** For every lane other than the winner with `Vy > 0`: `FF` then
   `RL` (down). On a wide group with few firing lanes this is a small handful of lanes; most of the
   bank gets no instruction at all.
4. **Stall response (conditional).** If `gather_abandon` updates have passed without the cycle
   completing, one of two things happens. With recruitment on: drive the highest-reading not-yet-won
   lane up (`FF` then `RH`) and mark it claimed, so the whole width stays live. With recruitment off
   (`abandon_action="reset"`): just clear the `won` bits and the counter — no lane is rewarded — so
   exclusion releases and the lanes that still win keep sharpening while the rest fade. The reset
   path is what turns the group into a self-pruning codebook (see the README); in hardware it is the
   same clear the cycle already uses, triggered by the counter instead of the `AND`-tree.

Every forward instruction in steps 2–4 is issued with its reverse partner, so the drive is balanced
by construction — no path leaves an `FF` unpaired.

## The one bit of new state: the cycle

The supervised RankCut needs no per-lane memory beyond the lanes themselves. The unsupervised rule
adds exactly one bit per lane — `won_i`, whether lane `i` has won a read since the last cycle reset —
plus a small shared counter:

- **exclusion** reads `won_w` on the winner: if the winner already won this cycle, the group issues
  no feedback on this update (steps 2–3 skipped). One bit read, one AND.
- **the cycle** completes when all `S` bits are set; a shared gate over the `won` bits detects
  `AND(won_0 … won_{S−1})`, clears every bit, and resets the counter.
- **recruitment** arms when the counter reaches `gather_abandon`; it grants the highest-reading
  unclaimed lane. The "highest-reading unclaimed" pick is the same swept-reference readout, masked
  by `won`.

So the whole hardware delta over RankCut is: one flip-flop per lane (`won_i`), one shared
`AND`-tree with a clear, one shared counter with a compare against `gather_abandon`, and the mask on
the readout. Everything else — the lane bank, the sub-threshold read, the balanced `FF`+reverse
drive, the swept-reference winner pick — is the RankCut hardware unchanged.

## Instructions used

| instruction | role |
|---|---|
| `FFLV` | sub-threshold decide read — non-disturbing, owes no reverse partner |
| `FF`   | the read half of a balanced correction |
| `RH`   | reverse feedback, drive the lane up (reward) |
| `RL`   | reverse feedback, drive the lane down (depress) |

Voltages are plain `V`; the read uses the low read voltage so a decision never writes.

## What is shared with RankCut

One routine, label optional. RankCut (supervised) drives the label's lane up and the confident-wrong
lanes down; the basis encoder (unsupervised) drives the read winner up and the other fired lanes
down. The read, the balanced drive, the depress of the fired lanes, and the swept-reference readout
are identical. A single hardware block serves both by choosing the target — the label's lane, or the
winner — and the basis encoder adds only the per-lane cycle bit.
