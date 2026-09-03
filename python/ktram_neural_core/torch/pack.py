"""The frozen weight pack — reading and writing the existing generator export format.

One layout, three consumers (the oracle's export validation, the browser Emu, these torch
modules): per section ("encoder" / "decoder" / "classifier") an int8 (diff, mag) pair with
one global symmetric scale each, plus the factored read-noise coefficients. Nothing new is
invented here; this file just moves that layout in and out of torch modules.

Two carriers of the same pack:
  - ``<base>.npz``       the numpy-native form (arrays + scales + noise keys) — the one
                         ``to_pack`` writes and ``from_pack`` prefers.
  - ``<base>.json/.bin`` the browser form (manifest + one packed int8 blob) — read-only here.
"""

import json
import pathlib
import warnings

import numpy as np

# Section name -> npz key prefix, and the packed-blob order of generator.py's export.
_PREFIX = {"encoder": "enc", "decoder": "dec", "classifier": "clf"}
_BLOB_ORDER = ["encoder", "decoder", "classifier"]
_NOISE_KEYS = ["a_thermal_unit", "a_flicker_unit", "sqrt_ref_m", "flicker_ln_ref",
               "ref_pw", "read_pw", "v_fflv", "v_cmp"]

# Keys a pack written before the comparator term existed cannot carry. Always written now; on
# read they are optional, and a pack missing them loads at the emulator's default (see
# _warn_pre_comparator). The browser emulator reads this block by key and does not model the
# comparator, so the new key is inert there — but no existing key is renamed or removed.
_POST_COMPARATOR_KEYS = ("v_cmp",)


def _warn_pre_comparator(noise):
    """A pack written before the comparator term has no level in it, so it loads at the
    emulator's default rather than at zero: the comparator is part of the read model, not part
    of the artifact, and an old pack was always read by SOME comparator. Say so once, because a
    silently applied default is how a number moves without anyone choosing it."""
    if noise and not any(k in noise for k in _POST_COMPARATOR_KEYS):
        warnings.warn(
            "pack has no comparator noise level (written before the term existed); "
            "loading at the emulator's default comparator. Pass v_cmp explicitly, or "
            "sample_read(..., comparator=False), to pin the device-only read.",
            stacklevel=3)


def _base(path):
    p = pathlib.Path(path)
    return p.with_suffix("") if p.suffix in (".npz", ".json", ".bin") else p


def quantize(a):
    """int8 with one global symmetric scale; returns (int8 array, real-units-per-step).
    Bit-identical to the generator export's ``_quantize`` (numpy round, half-to-even)."""
    s = float(np.abs(a).max()) or 1.0
    q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
    return q, s / 127.0


def load_section(path, section):
    """One section of a pack: (qdiff int8, qmag int8, diff_scale, mag_scale, noise dict).
    Prefers ``<base>.npz``; falls back to the browser's ``.json`` + ``.bin`` pair."""
    if section not in _PREFIX:
        raise KeyError(f"unknown pack section {section!r}; valid: {sorted(_PREFIX)}")
    base = _base(path)
    npz_path = base.with_suffix(".npz")
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=False)
        p = _PREFIX[section]
        noise = {k: float(data[f"noise_{k}"]) for k in _NOISE_KEYS
                 if f"noise_{k}" in data.files}
        _warn_pre_comparator(noise)
        return (data[f"{p}_diff"], data[f"{p}_mag"],
                float(data[f"{p}_diff_scale"]), float(data[f"{p}_mag_scale"]), noise)

    json_path = base.with_suffix(".json")
    bin_path = base.with_suffix(".bin")
    if not (json_path.exists() and bin_path.exists()):
        raise FileNotFoundError(f"no pack at {base} (.npz or .json/.bin)")
    manifest = json.loads(json_path.read_text())
    blob = np.frombuffer(bin_path.read_bytes(), dtype=np.int8)
    offset = 0
    for name in _BLOB_ORDER:
        shape = manifest[name]["shape"]
        n = int(np.prod(shape))
        diff = blob[offset:offset + n].reshape(shape)
        mag = blob[offset + n:offset + 2 * n].reshape(shape)
        offset += 2 * n
        if name == section:
            noise = dict(manifest.get("noise", {}))
            _warn_pre_comparator(noise)
            return (diff.copy(), mag.copy(), float(manifest[name]["diff_scale"]),
                    float(manifest[name]["mag_scale"]), noise)
    raise AssertionError("unreachable")


def save_section(path, section, diff, mag, noise=None):
    """Quantize and write one section into ``<base>.npz``, creating the file or updating the
    section in place (the other sections' keys are preserved)."""
    if section not in _PREFIX:
        raise KeyError(f"unknown pack section {section!r}; valid: {sorted(_PREFIX)}")
    base = _base(path)
    npz_path = base.with_suffix(".npz")
    existing = {}
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            existing = {k: data[k] for k in data.files}
    p = _PREFIX[section]
    qd, ds = quantize(np.asarray(diff, dtype=np.float32))
    qm, ms = quantize(np.asarray(mag, dtype=np.float32))
    existing[f"{p}_diff"] = qd
    existing[f"{p}_mag"] = qm
    existing[f"{p}_diff_scale"] = ds
    existing[f"{p}_mag_scale"] = ms
    if noise is not None:
        for k, v in noise.state().items():
            existing[f"noise_{k}"] = v
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **existing)
