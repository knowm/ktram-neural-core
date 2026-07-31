"""Thermal pattern generator — standalone interactive widget (Chapter 6b §10).

A self-contained reimplementation of the kT-RAM read (the same one the browser demo will port): it
depends only on numpy + matplotlib (+ sklearn for seed images), NOT on ktram_neural_core. It loads
the exported weights (`figures/generator-weights.npz` — int8 diff + mag + read-noise coefficients,
written by generator.py) and reproduces the emulator read exactly:

    TwoOne divider:  top = sum(diff at active), bottom = sum(mag at active), y_clean = top / bottom
    read noise:      y = clip(y_clean + T * sigma_unit(bottom, y_clean) * N(0,1), -1, 1)

Temperature T is the encoder Cores' read_noise gain, so the noise comes from the same read_sample law
the emulator uses (thermal + flicker in quadrature), scaled linearly by T. At T=0 the read is the
sharp argmax and the loop settles to a fixed point; raise T and the WTA winner is sampled, so the
generated image wanders between garment attractors. Decode is always sharp.

Run it:

    python generator_widget.py              # the interactive widget (needs a GUI backend)
    python generator_widget.py selftest     # headless: check it settles / explores, dump a few PNGs

Controls: temperature slider; New (random binary seed); Fashion (a dataset image); Random AAT (a
random code — one random winner per group — decoded straight to an image); Step; Play/Pause; Anneal
(ramp T -> 0 and snap to an attractor); Codebook (toggle a panel of a patch group's 64 learned
features — or click an AAT cell). The header shows the supervised label read-out's prediction.
"""

import sys
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WEIGHTS = HERE / "figures" / "generator-weights.npz"
MANIFEST = HERE / "figures" / "generator-weights.json"


# ---------------------------------------------------------------------------
# The standalone emulator — the exact kT-RAM read, from the exported weights.
# ---------------------------------------------------------------------------

class ThermalEmulator:
    def __init__(self, npz=WEIGHTS, manifest=MANIFEST):
        d = np.load(npz)
        m = json.loads(pathlib.Path(manifest).read_text())
        self.m = m
        self.SIDE = m["side"]
        self.LEVELS = m["levels"]
        self.BASIS = m["basis"]
        self.N_GROUPS = m["n_groups"]
        self.PATCH = m["patch"]
        self.STARTS = m["starts"]
        self.N_PIXELS = m["n_pixels"]
        self.N_PIX_PATCH = self.PATCH * self.PATCH

        # int8 weights + real-unit scales
        self.enc_diff = d["enc_diff"].astype(np.float32) * float(d["enc_diff_scale"])   # [16,32,49,2]
        self.enc_mag = d["enc_mag"].astype(np.float32) * float(d["enc_mag_scale"])
        self.dec_diff = d["dec_diff"].astype(np.float32) * float(d["dec_diff_scale"])   # [1568,16,32]
        self.dec_mag = d["dec_mag"].astype(np.float32) * float(d["dec_mag_scale"])

        # supervised label read-out (optional — present once the classifier is exported)
        self.labels = list(m.get("classifier", {}).get("labels", [])) or None
        if "clf_diff" in d:
            self.clf_diff = d["clf_diff"].astype(np.float32) * float(d["clf_diff_scale"])  # [10,16,32]
            self.clf_mag = d["clf_mag"].astype(np.float32) * float(d["clf_mag_scale"])
        else:
            self.clf_diff = self.clf_mag = None

        # read-noise coefficients (temperature is a linear multiplier on these)
        n = m["noise"]
        self.a_th = n["a_thermal_unit"]
        self.a_fl = n["a_flicker_unit"]
        self.sqrt_ref_m = n["sqrt_ref_m"]
        self.flicker_ln_ref = n["flicker_ln_ref"]
        self.ref_pw = n["ref_pw"]
        self.read_pw = n["read_pw"]
        self.v_fflv = n["v_fflv"]
        # pulse-width bandwidth factors (constant here since read_pw == ref_pw)
        self.bw_th = np.sqrt(self.ref_pw / self.read_pw)
        ln_band = self.flicker_ln_ref + np.log(self.ref_pw / self.read_pw)
        self.bw_fl = np.sqrt(ln_band / self.flicker_ln_ref) if ln_band > 0 else 0.0

        grid = np.arange(self.SIDE * self.SIDE).reshape(self.SIDE, self.SIDE)
        self.patch_idx = np.array([grid[r:r + self.PATCH, c:c + self.PATCH].ravel()
                                   for r in self.STARTS for c in self.STARTS])
        self.hot = m.get("hot_read_noise", 0.5)

    def _sigma(self, m, y):
        """Per-lane read-noise sigma at read_noise gain = 1 (the emulator's read_sample law). m and y
        are per-lane arrays; multiply the result by the temperature to get the actual sigma."""
        m = np.maximum(m, 1e-9)
        f_m = self.sqrt_ref_m / np.sqrt(m)
        st = self.a_th * f_m / self.v_fflv * self.bw_th
        sf = self.a_fl * (1.0 - y * y) * f_m * self.bw_fl
        return np.sqrt(st * st + sf * sf)

    def encode(self, image, temperature=0.0, rng=None):
        """image: int8 [784] binary levels -> int8 [16] winner indices. At T>0 each lane's read
        carries emulator read noise, so the WTA winner is sampled."""
        aat = np.empty(self.N_GROUPS, dtype=np.int8)
        sp = np.arange(self.N_PIX_PATCH)
        for g in range(self.N_GROUPS):
            lv = image[self.patch_idx[g]]                      # [49] levels 0/1
            top = self.enc_diff[g][:, sp, lv].sum(axis=1)      # [32]
            bot = self.enc_mag[g][:, sp, lv].sum(axis=1)       # [32]
            y = top / np.where(bot != 0, bot, 1.0)
            if temperature > 0.0 and rng is not None:
                y = np.clip(y + temperature * self._sigma(bot, y) * rng.standard_normal(self.BASIS),
                            -1.0, 1.0)
            aat[g] = int(np.argmax(y))
        return aat

    def decode(self, aat):
        """aat: int [16] -> int8 [784] levels. Sharp divider read, argmax per pixel."""
        sp = np.arange(self.N_GROUPS)
        idx = np.asarray(aat, dtype=np.intp)
        top = self.dec_diff[:, sp, idx].sum(axis=1)            # [1568]
        bot = self.dec_mag[:, sp, idx].sum(axis=1)
        y = top / np.where(bot != 0, bot, 1.0)
        return y.reshape(self.N_PIXELS, self.LEVELS).argmax(axis=1).astype(np.int8)

    def codebook(self, g):
        """Group g's learned codebook: [BASIS, PATCH, PATCH]. Each lane's patch prototype is its
        ink-channel differential (Ga-Gb on the ink level) reshaped to the patch — bright where the
        lane expects ink. Pruned/dead lanes read ~0 (they never won, so stayed near init)."""
        ink = self.LEVELS - 1                                  # ink level (=1 for binary)
        return self.enc_diff[g][:, :, ink].reshape(self.BASIS, self.PATCH, self.PATCH)

    def classify(self, aat):
        """The supervised label read-out over the same AAT: sharp divider read, argmax over classes.
        Returns (label_index, label_name), or (None, None) if no classifier was exported."""
        if self.clf_diff is None:
            return None, None
        sp = np.arange(self.N_GROUPS)
        idx = np.asarray(aat, dtype=np.intp)
        top = self.clf_diff[:, sp, idx].sum(axis=1)
        bot = self.clf_mag[:, sp, idx].sum(axis=1)
        y = top / np.where(bot != 0, bot, 1.0)
        k = int(np.argmax(y))
        return k, (self.labels[k] if self.labels else str(k))

    def step(self, image, temperature, rng):
        return self.decode(self.encode(image, temperature, rng))


# ---------------------------------------------------------------------------
# Seeds.
# ---------------------------------------------------------------------------

def fashion_seeds(n, side=28, seed=0):
    """n Fashion-MNIST images, binarized per-image-mean (matching the training encoding)."""
    from sklearn.datasets import fetch_openml
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(np.float32)
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))[:n]]
    return (X > X.mean(axis=1, keepdims=True)).astype(np.int8)


def random_seed(emu, rng):
    """A random binary blob seed (ink in a rough central region)."""
    img = (rng.random(emu.N_PIXELS) < 0.25).astype(np.int8)
    return img


def random_aat_seed(emu, rng):
    """A random AAT decoded to an image: one random winner index per patch group, run through the
    decoder. A random point in the code space the encoder learned, made visible — not a random image,
    a random *code*. The loop then relaxes it toward the nearest garment the two halves agree on."""
    aat = rng.integers(0, emu.BASIS, size=emu.N_GROUPS).astype(np.int8)
    return emu.decode(aat)


# ---------------------------------------------------------------------------
# Headless self-test — confirm the standalone loop settles and explores.
# ---------------------------------------------------------------------------

def selftest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emu = ThermalEmulator()
    print("manifest validation (Core agreement at export):", emu.m["validation"], flush=True)
    seeds = fashion_seeds(6, side=emu.SIDE, seed=1)
    rng = np.random.default_rng(0)

    # settle at T=0
    print("\n[settle] T=0 fixed points:", flush=True)
    finals = []
    for j in range(len(seeds)):
        img = seeds[j].copy()
        prev = None
        for t in range(20):
            img = emu.step(img, 0.0, rng)
            if prev is not None and (img == prev).all():
                break
            prev = img.copy()
        finals.append(img)
        print(f"  seed {j}: settled step {t}, ink {(img>0).mean():.2f}", flush=True)

    # explore across temperature
    print("\n[explore] distinct states over 30 steps per T:", flush=True)
    for T in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]:
        img = seeds[0].copy()
        states = set()
        for _ in range(30):
            img = emu.step(img, T, rng)
            states.add(img.tobytes())
        print(f"  T={T:>4}: {len(states):>2} distinct, ink {(img>0).mean():.2f}", flush=True)

    # dump a filmstrip: anneal hot -> 0
    img = seeds[3].copy()
    frames = [img.copy()]
    temps = list(np.linspace(emu.hot, 0.0, 24)) + [0.0] * 6
    for T in temps:
        img = emu.step(img, T, rng)
        frames.append(img.copy())
    fig, axes = plt.subplots(1, 10, figsize=(16, 1.8))
    idx = np.linspace(0, len(frames) - 1, 10).astype(int)
    for ax, f in zip(axes, idx):
        ax.imshow(frames[f].reshape(emu.SIDE, emu.SIDE), cmap="magma", vmin=0, vmax=emu.LEVELS - 1)
        ax.axis("off")
    fig.suptitle("standalone widget selftest — anneal hot -> 0", fontsize=10)
    out = HERE / "figures" / "widget-selftest.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}", flush=True)


# ---------------------------------------------------------------------------
# The interactive widget.
# ---------------------------------------------------------------------------

def run_widget():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patheffects as pe

    # green phosphor terminal palette
    GREEN = "#39ff14"          # bright phosphor — text, accents, winner box
    GREEN_DIM = "#1f9e33"      # dim green — secondary text, borders
    PANEL = "#050805"          # near-black panel fill
    phosphor = LinearSegmentedColormap.from_list("phosphor", ["#000000", "#0b3d0b", GREEN])

    def greenify(a):
        """Black facecolor + dim-green spines on an axes — the terminal look."""
        a.set_facecolor("black")
        for sp in a.spines.values():
            sp.set_color(GREEN_DIM)

    emu = ThermalEmulator()
    rng = np.random.default_rng()
    seeds = fashion_seeds(200, side=emu.SIDE, seed=7)
    seed_i = {"n": 0}
    side = int(round(emu.N_GROUPS ** 0.5))                 # 4
    state = {"img": seeds[0].copy(), "playing": False, "anneal": None,
             "aat": emu.encode(seeds[0], 0.0), "cb_group": 5, "cb_on": False}

    fig = plt.figure(figsize=(5.6, 8.6), facecolor="black")
    fig.canvas.manager.set_window_title("kT-RAM thermal generator")

    # header (never clipped): predicted label big, T / ink small under it
    head = fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=15, weight="bold",
                    color=GREEN, family="monospace")
    sub = fig.text(0.5, 0.935, "", ha="center", va="top", fontsize=10, color=GREEN_DIM,
                   family="monospace")

    ax = fig.add_axes([0.06, 0.375, 0.88, 0.52], facecolor="black")   # the image
    im = ax.imshow(state["img"].reshape(emu.SIDE, emu.SIDE), cmap=phosphor,
                   vmin=0, vmax=emu.LEVELS - 1, interpolation="nearest")
    ax.axis("off")

    # the integer AAT as a 4x4 grid (each cell = one patch group's winner index 0..BASIS-1).
    # Click a cell to open that group's codebook.
    ax_aat = fig.add_axes([0.335, 0.185, 0.33, 0.155])
    greenify(ax_aat)
    aat_im = ax_aat.imshow(np.zeros((side, side)), cmap=phosphor, vmin=0, vmax=emu.BASIS - 1)
    ax_aat.set_xticks([]); ax_aat.set_yticks([])
    ax_aat.set_title(f"AAT — {emu.N_GROUPS} patch winners (0–{emu.BASIS - 1}) · click a cell",
                     fontsize=8.5, color=GREEN_DIM, family="monospace")
    aat_txt = [[ax_aat.text(j, i, "", ha="center", va="center", fontsize=8, color=GREEN,
                            family="monospace",
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
                for j in range(side)] for i in range(side)]

    ax_T = fig.add_axes([0.17, 0.125, 0.66, 0.03])
    greenify(ax_T)
    s_T = Slider(ax_T, "temp\n(read_noise)", 0.0, 0.6, valinit=0.0, valstep=0.005,
                 color=GREEN, initcolor="none")
    s_T.label.set_color(GREEN_DIM); s_T.label.set_family("monospace")
    s_T.valtext.set_color(GREEN); s_T.valtext.set_family("monospace")
    if hasattr(s_T, "track"):
        s_T.track.set_facecolor(PANEL); s_T.track.set_edgecolor(GREEN_DIM)
    if hasattr(s_T, "_handle"):
        s_T._handle.set_markerfacecolor(GREEN); s_T._handle.set_markeredgecolor(GREEN)

    def draw():
        img = state["img"]
        im.set_data(img.reshape(emu.SIDE, emu.SIDE))
        aat = emu.encode(img, temperature=0.0)             # sharp code of the shown image
        state["aat"] = aat
        grid = aat.reshape(side, side)
        aat_im.set_data(grid)
        for i in range(side):
            for j in range(side):
                aat_txt[i][j].set_text(str(int(grid[i, j])))
        _, name = emu.classify(aat)
        mode = "annealing" if state["anneal"] is not None else \
               ("playing" if state["playing"] else "paused")
        head.set_text(name if name else "kT-RAM generator")
        sub.set_text(f"T = {s_T.val:.3f}    ink = {(img > 0).mean():.2f}    [{mode}]")
        refresh_codebook()                                 # keep the codebook's winner box in sync
        fig.canvas.draw_idle()

    def do_step():
        if state["anneal"] is not None:
            k, n = state["anneal"]
            s_T.set_val(emu.hot * (1 - k / n))             # ramp the slider down visibly
            state["img"] = emu.step(state["img"], s_T.val, rng)
            state["anneal"] = None if k + 1 > n else (k + 1, n)
            if state["anneal"] is None:
                s_T.set_val(0.0)
        else:
            state["img"] = emu.step(state["img"], s_T.val, rng)
        draw()

    timer = fig.canvas.new_timer(interval=110)
    timer.add_callback(lambda: do_step() if (state["playing"] or state["anneal"] is not None) else None)
    timer.start()

    # ---- codebook panel (SAME window, toggled — a second native window segfaults the macosx
    #      backend when closed while the timer runs, so we overlay the image axes instead) ------
    ncb = int(np.ceil(emu.BASIS ** 0.5))                   # 8 columns for 64 basis
    P, gap = emu.PATCH, 1
    Wtile = ncb * P + (ncb - 1) * gap
    ax_cb = fig.add_axes([0.06, 0.375, 0.88, 0.52], facecolor="black")   # overlays the image rect
    ax_cb.set_xticks([]); ax_cb.set_yticks([]); ax_cb.set_visible(False)
    cb_im = ax_cb.imshow(np.zeros((Wtile, Wtile)), cmap=phosphor, vmin=0, vmax=1,
                         interpolation="nearest")
    cb_box = Rectangle((0, 0), P, P, fill=False, edgecolor=GREEN, lw=2.0)
    ax_cb.add_patch(cb_box)

    def refresh_codebook():
        if not state["cb_on"]:
            return
        g = state["cb_group"]
        cb = np.maximum(emu.codebook(g), 0.0)              # relu: bright = expects ink
        canvas = np.full((Wtile, Wtile), np.nan)
        for k in range(emu.BASIS):
            r, c = divmod(k, ncb)
            canvas[r * (P + gap):r * (P + gap) + P, c * (P + gap):c * (P + gap) + P] = cb[k]
        cb_im.set_data(canvas); cb_im.set_clim(0, float(np.percentile(cb, 99.5)) or 1.0)
        win = int(state["aat"][g])                         # current winner for this group
        r, c = divmod(win, ncb)
        cb_box.set_xy((c * (P + gap) - 0.5, r * (P + gap) - 0.5))
        cb_box.set_width(P); cb_box.set_height(P)
        rr, cc = divmod(g, side)
        head.set_text(f"codebook · group {g}")
        sub.set_text(f"patch row {rr}, col {cc}   ·   {emu.BASIS} learned {P}×{P} features   "
                     f"·   winner = {win}")

    def _show_codebook(on):
        state["cb_on"] = on
        ax.set_visible(not on)                             # image hidden while codebook shows
        ax_cb.set_visible(on)

    def toggle_codebook(_=None):
        _show_codebook(not state["cb_on"])
        draw()

    def on_aat_click(event):
        if event.inaxes is ax_aat and event.xdata is not None:
            c, r = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= r < side and 0 <= c < side:
                state["cb_group"] = r * side + c           # pick the group; show its codebook
                _show_codebook(True)
                draw()
    fig.canvas.mpl_connect("button_press_event", on_aat_click)

    # ---- main controls --------------------------------------------------------
    def add_button(x, w, label, cb):
        b = Button(fig.add_axes([x, 0.035, w, 0.06]), label, color=PANEL, hovercolor="#0f2a0f")
        b.label.set_color(GREEN); b.label.set_family("monospace"); b.label.set_fontsize(9)
        for sp in b.ax.spines.values():
            sp.set_color(GREEN_DIM)
        b.on_clicked(cb)
        return b

    def on_new(_):
        state["img"] = random_seed(emu, rng); state["anneal"] = None; draw()

    def on_random_aat(_):
        state["img"] = random_aat_seed(emu, rng); state["anneal"] = None; draw()

    def on_fashion(_):
        seed_i["n"] = (seed_i["n"] + 1) % len(seeds)
        state["img"] = seeds[seed_i["n"]].copy(); state["anneal"] = None; draw()

    def on_play(_):
        state["playing"] = not state["playing"]; draw()

    def on_anneal(_):
        state["playing"] = False; state["anneal"] = (0, 30)

    fig._buttons = [
        add_button(0.020, 0.075, "New", on_new),
        add_button(0.126, 0.110, "Fashion", on_fashion),
        add_button(0.267, 0.145, "Random AAT", on_random_aat),
        add_button(0.443, 0.080, "Step", lambda _: do_step()),
        add_button(0.554, 0.145, "Play/Pause", on_play),
        add_button(0.730, 0.100, "Anneal", on_anneal),
        add_button(0.861, 0.120, "Codebook", toggle_codebook),
    ]
    s_T.on_changed(lambda _v: draw())                      # slider move updates the readout live
    draw()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        run_widget()
