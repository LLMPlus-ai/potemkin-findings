"""21 - Proposal figure: the five Null-World Calibration steps, each with its analytical
output on the application (four greys; greyscale-legible; same style as 15_figures.py).

Fig 6  fig6_steps.pdf   strip of the five steps + six panels:
                        (1) design space multiplies to the registered specifications
                        (2) specification curve of H1 on the real data; where the loop stopped
                        (3) null worlds: largest H1 t over the search; package statistic (Potemkin rate)
                        (4) boundary conditions and mechanisms (moderators): p under three world constructions,
                            for the package t and the screening t
                        (5) planted worlds: winner's curse and power
                        (Output) each statistic with its own search-adjusted p; familywise over all ten moderators
Labels are written for readers without a statistics background (plain-language pass, 2026-10-21).
Inputs: registry/design_space.yaml, intermediate/10_loop_real.json, intermediate/10_spec_curve.csv,
        intermediate/14_null_summary.csv, tables/nwc.json, tables/validity.json (scripts/25)
"""
import sys, json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd, yaml
from plotstyle import setup, GREYS, plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

F = ROOT / "figures"
TITLE_PAD = 20          # bottom-row titles sit above two-row legends, on one baseline
PLAIN_DIM = {"treatments": "exposures"}
MODLAB = {"pyramid_z": "hierarchy shape", "remote_z": "digitizable work", "firm_age_z": "organizational age"}


def mod_t(pkg, m):
    if pkg is None or m not in pkg["moderators"]:
        return -np.inf
    return pkg["t_mods"][pkg["moderators"].index(m)]


def strip(ax):
    """Five numbered step boxes with arrows, pure black and white. The boxes fill the axes exactly
    (outer edges at x = 0 and x = 1, nothing clipped), so the strip spans whatever range the axes are given.
    Each step is named by one verb; its gloss names what the verb acts on."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [("1", "Freeze", "lock every choice and the\nsearch rules before results", 1.0),
             ("2", "Run", "the AI's coded search\non the real data", 1.0),
             ("3", "Replay", "the same search on shuffled\nworlds with no true effect", 1.0),
             ("4", "Test", "boundary conditions and\nmechanisms, trait by trait", 1.0),
             ("5", "Plant", "add known effects: how often\nfound, how much exaggerated", 1.0)]
    n, gap = len(steps), 0.02
    unit = (1 - (n - 1) * gap) / sum(r for *_, r in steps)
    x0 = 0.0
    for i, (num, verb, gloss, rel) in enumerate(steps):
        w = rel * unit
        ax.add_patch(FancyBboxPatch((x0, 0.03), w, 0.94, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="#FFFFFF", ec="#000000", lw=0.7, clip_on=False))
        ax.text(x0 + 0.017, 0.70, num, fontsize=6.6, weight="bold", color="#FFFFFF", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.22", fc="#000000", ec="#000000", lw=0), clip_on=False)
        ax.text(x0 + 0.034, 0.70, verb, fontsize=7.5, weight="bold", color="#000000", ha="left", va="center", linespacing=1.05)
        ax.text(x0 + w / 2, 0.25, gloss, fontsize=5.9, color="#000000", ha="center", va="center", linespacing=1.15)
        x0 += w + gap
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x0 - gap + 0.002, 0.5), (x0 - 0.002, 0.5), arrowstyle="-|>",
                                         mutation_scale=7, lw=0.8, color="#000000", clip_on=False))


def panel_freeze(ax, reg, hash_, frozen):
    dims = [("outcomes", len(reg["outcomes"])), ("treatments", len(reg["treatments"])),
            ("controls", len(reg["controls"])), ("fixed effects", len(reg["fixed_effects"])),
            ("samples", len(reg["samples"])), ("transforms", len(reg["transforms"])),
            ("clustering", len(reg["clustering"]))]
    cum = np.cumprod([n for _, n in dims])
    y = np.arange(len(dims))[::-1]
    ax.barh(y, cum - 1, left=1, height=0.62, color=GREYS[2], edgecolor=GREYS[0], lw=0.5)
    for yi, v in zip(y, cum):
        ax.text(v * 1.18, yi, f"{v:,}", va="center", fontsize=6.2)
    ax.set_xscale("log"); ax.set_xlim(1, 4000); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks(y); ax.set_yticklabels([f"{PLAIN_DIM.get(d, d)} ×{n}" for d, n in dims])
    ax.set_ylim(-0.6, len(dims) - 0.4)
    ax.set_xlabel("Number of analyses (cumulative)")
    ax.text(0.98, 0.97, f"+ {len(reg['moderators'])} moderators,\nup to 10 sets of 3\nper analysis\nSHA-256 {hash_[:8]}…\n"
            f"locked {frozen}", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, linespacing=1.2)
    ax.set_title(f"1 Freeze: {cum[-1]:,} registered analyses")


def panel_run(ax, sc, real, pkg):
    s = sc.sort_values("t_h1").reset_index(drop=True)
    ax.scatter(s.index, s.t_h1, s=2.2, color=GREYS[1], linewidths=0)
    ax.axhline(1.96, color=GREYS[0], ls="--", lw=0.7); ax.axhline(0, color=GREYS[2], lw=0.5)
    keys = ["outcome", "treatment", "controls", "fe", "sample", "transform", "cluster"]
    m = np.ones(len(s), bool)
    for k in keys:
        m &= (s[k] == pkg[k]).to_numpy()
    if m.any():
        i = int(np.flatnonzero(m)[0])
        ax.scatter([i], [s.t_h1[i]], s=28, facecolors="none", edgecolors=GREYS[0], lw=0.9, zorder=5)
        ax.annotate(f"analysis behind the\nreported results\n(H1 alone: t = {s.t_h1[i]:.1f})", (i, s.t_h1[i]), xytext=(150, 6.6),
                    textcoords="data", fontsize=5.8, ha="center", va="center", linespacing=1.1,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=GREYS[0], shrinkB=3))
    else:
        print("warning: package base specification not found in spec curve", pkg)
    share = 100 * (s.t_h1 >= 1.96).mean()
    ax.text(0.03, 0.97, f"{share:.0f}% of analyses\nsignificant (t ≥ 1.96)", transform=ax.transAxes, ha="left", va="top", fontsize=5.9)
    tm = sorted(pkg["t_mods"], reverse=True)
    ax.text(0.97, 0.03, f"search stops at model {real['U']['n_models_to_first']:,}:\nall four significant, H1 t = {pkg['t_h1']:.1f},\n"
            f"moderators t = {tm[0]:.1f}, {tm[1]:.1f}, {tm[2]:.1f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, linespacing=1.15)
    ax.set_ylim(-4.8, max(s.t_h1.max() + 0.6, 12.8))
    ax.set_xlabel("Analyses, ranked by t"); ax.set_ylabel("H1 t (in the predicted direction)")
    ax.set_xlim(-10, len(s) + 10); ax.set_title("2 Run: the search on the real data")


def panel_replay(ax_top, ax_bot, null, nwc):
    B = len(null)
    # top: largest H1 statistic over the search
    h = null.h1_max.to_numpy(); r = nwc["real"]["h1_max"]
    ax_top.hist(h, bins=24, color=GREYS[3], edgecolor=GREYS[2], lw=0.4)
    ax_top.axvline(r, color=GREYS[0], lw=1.1); ax_top.set_ylim(0, ax_top.get_ylim()[1] * 1.7)
    ph1 = nwc["p_nwc_h1_max"]; rel = "≤" if abs(ph1 - 1.0 / (B + 1)) < 1e-12 else "="
    ax_top.text(r - 0.15, ax_top.get_ylim()[1] * 0.95, f"real {r:.1f}\np {rel} {ph1:.3f}", ha="right", va="top", fontsize=5.9)
    ax_top.text(0.02, 0.95, "best H1 t\nfound by the search", transform=ax_top.transAxes, ha="left", va="top", fontsize=5.9)
    ax_top.set_xlim(min(h.min(), 5) - 0.3, r + 0.6); ax_top.set_ylabel("Worlds")
    ax_top.set_title(f"3 Replay: {B:,} null worlds")
    # bottom: package statistic, tail above 1.96 = Potemkin rate
    T = null.U_T.replace(-np.inf, np.nan).dropna().to_numpy(); rT = nwc["real"]["U_T"]
    edges = 1.96 + 0.2 * np.arange(-21, 11)
    cnt, _ = np.histogram(T, bins=edges)
    for c, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
        ax_bot.bar(e0, c, width=e1 - e0, align="edge", color=GREYS[0] if e0 >= 1.96 - 1e-9 else GREYS[3], edgecolor=GREYS[2], lw=0.4)
    ax_bot.axvline(rT, color=GREYS[0], lw=1.1); ax_bot.set_ylim(0, cnt.max() * 1.8)
    rate = 100 * nwc["potemkin_rates"]["U (registered signs)"]["rate"]
    ax_bot.text(rT - 0.1, cnt.max() * 1.8 * 0.95, f"false successes {rate:.1f}% (dark)\nreal {rT:.2f}, p = {nwc['p_nwc_package_U']:.3f}",
                ha="right", va="top", fontsize=5.9, linespacing=1.15)
    ax_bot.set_xlim(edges[0], edges[-1]); ax_bot.set_ylabel("Worlds"); ax_bot.set_xlabel("Weakest t of the four hypotheses")
    print("check: share(U_T >= 1.96) =", float((null.U_T >= 1.96).mean()), " success rate =", float(null.U_success.mean()))


def panel_moderators(ax, nwc, pkg, val):
    """For each reported moderator: search-adjusted p of its t in the reported package (-inf when the loop does not
    report it: strength and selection) under three world constructions (circle = permutation, square = conditional
    permutation, triangle = null-model worlds), and of its largest screening t in permuted worlds (open circle; defined
    in every world, strength only). scripts/22, scripts/25."""
    d, c = val["decomposition"], val["constructions"]
    mods = [m for m in ["pyramid_z", "remote_z", "firm_age_z"] if m in d]
    hyp = {m: f"H{i + 2}" for i, m in enumerate(["pyramid_z", "remote_z", "firm_age_z"])}
    y = np.arange(len(mods))[::-1]
    marks = [("permB", "p_sel", "o", GREYS[0], GREYS[0], 0.21), ("residB", "p_sel", "s", GREYS[1], GREYS[1], 0.07),
             ("fl", "p_sel", "^", GREYS[0], GREYS[0], -0.07), ("permB", "p_free", "o", GREYS[0], "#FFFFFF", -0.21)]
    for cc, st, mk, col, mfc, dy in marks:
        if not all(c.get(cc, {}).get(m, {}).get("B", 0) >= 500 for m in mods):
            continue
        ax.plot([c[cc][m][st] for m in mods], y + dy, ls="none", marker=mk, ms=3.6, color=col, mfc=mfc, mec=col, mew=0.8, zorder=5)
    ax.axvline(0.05, color=GREYS[0], ls="--", lw=0.7)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1.6); ax.set_xticks([1e-3, 1e-2, 0.05, 1]); ax.set_xticklabels(["0.001", "0.01", "0.05", "1"])
    ax.set_yticks(y); ax.set_yticklabels([f"{hyp[m]} {MODLAB[m]}\nreported in {100 * d[m]['share_selected']:.1f}%\nof its shuffled worlds"
                                          for m in mods], fontsize=5.6, linespacing=1.05)
    ax.set_ylim(-0.6, len(mods) - 0.4)
    ax.set_xlabel("Search-adjusted p (log scale)")
    h = [Line2D([], [], ls="none", marker="o", ms=3.3, color=GREYS[0], label="simple shuffle"),
         Line2D([], [], ls="none", marker="^", ms=3.3, color=GREYS[0], label="rebuilt outcomes"),
         Line2D([], [], ls="none", marker="s", ms=3.3, color=GREYS[1], label="conditional shuffle"),
         Line2D([], [], ls="none", marker="o", ms=3.3, color=GREYS[0], mfc="#FFFFFF", label="best t in the search")]
    ax.legend(handles=h, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handletextpad=0.15,
              columnspacing=0.8, borderaxespad=0.1, fontsize=5.3, labelspacing=0.25)
    ax.set_title("4 Test: boundary conditions and mechanisms", pad=TITLE_PAD)


def panel_plant(ax, nwc):
    d = pd.DataFrame(nwc["planted"].values()).sort_values("beta_star")
    ax.plot(d.beta_star, d.winner_ratio_median, "-", color=GREYS[0], marker="o", ms=4)
    ax.axhline(1, color=GREYS[2], ls="--", lw=0.8)
    for b, r in zip(d.beta_star, d.winner_ratio_median):
        ax.text(b, r + 0.35, f"{r:.1f}×", ha="center", va="bottom", fontsize=6.2)
    ax.text(0.98, 0.03, "no exaggeration (ratio 1)", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, color=GREYS[1])
    pw = lambda c: f"{100 * d[c].min():.0f}" + ("" if d[c].min() == d[c].max() else f"–{100 * d[c].max():.0f}")
    ax.text(0.98, 0.95, f"power to find the effect:\nsearch {pw('U_power')}%, calibrated {pw('nwc_power')}%,\nBonferroni {pw('bonf_power')}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.9, linespacing=1.15)
    ax.set_xticks(d.beta_star.tolist()); ax.set_xlim(d.beta_star.min() - 0.01, d.beta_star.max() + 0.01)
    ax.set_ylim(0, max(d.winner_ratio_median.max() * 1.25, 2))
    ax.set_xlabel("Planted main effect (within-firm SD per SD of exposure)"); ax.set_ylabel("Reported ÷ true effect (median)")
    ax.set_title("5 Plant: power and exaggeration", pad=TITLE_PAD)


def panel_output(ax, nwc, pkg, val):
    """Each row pairs a statistic with ITS search-adjusted p. H1: the largest H1 t over the search (the statistic
    the H1 p ranks), not the H1 t inside the package. Package: the smallest of its four t. Moderators: the t in
    the package; open markers = familywise adjustment over all ten registered candidates (Holm; unreported = 1)."""
    d, fam = val["decomposition"], val.get("familywise")
    rows = [("H1 main effect,\nbest t in search", nwc["real"]["h1_max"], nwc["p_nwc_h1_max"], None, nwc["B_typeA"]),
            ("All four together,\nweakest t", nwc["real"]["U_T"], nwc["p_nwc_package_U"], None, nwc["B_typeA"])]
    short = {"pyramid_z": "H2 hierarchy\nshape, t", "remote_z": "H3 digitizable\nwork, t", "firm_age_z": "H4 organiz.\nage, t"}
    for m in ["pyramid_z", "remote_z", "firm_age_z"]:
        fw = val["constructions"]["permB"][m]["holm10"]          # Holm over all ten candidates (primary)
        rows.append((short[m], d[m]["t_obs"], d[m]["p_sel"], fw, d[m]["B"]))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, t, p, ph, B) in zip(y, rows):
        floor = 1.0 / (B + 1)
        if ph is not None and ph > p:
            ax.plot([p, ph], [yi, yi], color=GREYS[1], lw=0.8)
            ax.plot([ph], [yi], marker="o", ms=4.2, mfc="#FFFFFF", mec=GREYS[0], ls="none", zorder=4)
        ax.plot([p], [yi], marker="o", ms=4.2, color=GREYS[0], ls="none", zorder=5)
        txt = f"p ≤ {floor:.3f}" if abs(p - floor) < 1e-12 else f"p = {p:.3f}"
        ax.text(p, yi + 0.24, txt, ha="left" if p < 0.003 else "center", va="bottom", fontsize=5.6)
    ax.axvline(0.05, color=GREYS[0], ls="--", lw=0.7)
    ax.set_xscale("log"); ax.set_xlim(5e-4, 1.6); ax.set_xticks([1e-3, 1e-2, 0.05, 1]); ax.set_xticklabels(["0.001", "0.01", "0.05", "1"])
    ax.set_yticks(y); ax.set_yticklabels([f"{lab} = {t:.1f}" for lab, t, _, _, _ in rows], fontsize=5.6, linespacing=1.05)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.75)
    ax.legend(handles=[Line2D([], [], marker="o", ms=3.6, color=GREYS[0], ls="none", label="adjusted for the search"),
                       Line2D([], [], marker="o", ms=3.6, mfc="#FFFFFF", mec=GREYS[0], ls="none", label="also corrected for 10 candidates (Holm)")],
              frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, borderaxespad=0.1, handletextpad=0.2,
              fontsize=5.3, labelspacing=0.25)
    ax.set_xlabel("Search-adjusted p (log scale)"); ax.set_title("Output: the corrected report", pad=TITLE_PAD)


def main():
    setup()
    plt.rcParams.update({"font.size": 6.8, "axes.titlesize": 7.2, "axes.labelsize": 6.5, "xtick.labelsize": 6.1,
                         "ytick.labelsize": 6.1, "legend.fontsize": 5.9, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.titlepad": 3, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})
    nwc = json.loads((ROOT / "tables/nwc.json").read_text())
    val = json.loads((ROOT / "tables/validity.json").read_text())
    real = json.loads((ROOT / "intermediate/10_loop_real.json").read_text())
    reg = yaml.safe_load((ROOT / "registry/design_space.yaml").read_text())
    null = pd.read_csv(ROOT / "intermediate/14_null_summary.csv")
    sc = pd.read_csv(ROOT / "intermediate/10_spec_curve.csv")
    pkg = real["U"]["first_success"]
    fig = plt.figure(figsize=(6.5, 4.7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.78, wspace=0.62, left=0.095, right=0.995, top=0.78, bottom=0.075)
    panel_freeze(fig.add_subplot(gs[0, 0]), reg, nwc["registry_sha256"], reg.get("frozen_at", ""))
    panel_run(fig.add_subplot(gs[0, 1]), sc, real, pkg)
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.55)
    panel_replay(fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), null, nwc)
    panel_moderators(fig.add_subplot(gs[1, 0]), nwc, pkg, val)
    panel_plant(fig.add_subplot(gs[1, 1]), nwc)
    panel_output(fig.add_subplot(gs[1, 2]), nwc, pkg, val)
    # The step strip spans exactly the horizontal range of the panels (tick labels and legends included),
    # so its left and right edges line up with the outermost content below and the figure has one width.
    panels = list(fig.axes)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    ext = Bbox.union([a.get_tightbbox(r) for a in panels])
    fx = ext.transformed(fig.transFigure.inverted())
    sax = fig.add_axes([fx.x0, 0.855, fx.width, 0.14]); strip(sax)
    fig.canvas.draw()
    sb = sax.get_tightbbox(r)
    print(f"check: panels x = [{ext.x0:.1f}, {ext.x1:.1f}] px; strip x = [{sb.x0:.1f}, {sb.x1:.1f}] px")
    fig.savefig(F / "fig6_steps.pdf", bbox_inches="tight", pad_inches=0.02)
    if len(sys.argv) > 1:
        fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("figures/fig6_steps.pdf written")


if __name__ == "__main__":
    main()
