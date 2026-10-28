"""15 - Article and appendix figures (four greys; greyscale-legible).

Fig 1  fig1_workflow.pdf       replayable loop and null-world calibration (schematic)
Fig 2  fig2_potemkin.pdf       Potemkin rates by procedure in Type A null worlds
Fig 3  fig3_testbed.pdf        (a) event study; (b) real vs null H1 statistic;
                               (c) real vs null package statistic
Fig 4  fig4_conceptual.pdf     conceptual diagram of H1-H4 (registered signs)
Fig 5  fig5_planted.pdf        power and winner's curse in planted worlds
Fig A  figA_spec_curve.pdf     specification curve of H1 over the 648 base specifications
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from plotstyle import setup, GREYS, LINES, MARKERS, plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

F = ROOT / "figures"


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=GREYS[0], lw=0.8, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, ls="-", color=GREYS[0], lw=0.9, text=None, tx=0, ty=0, fs=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=lw, ls=ls, color=color))
    if text:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, fontsize=fs, ha="center", va="center")


def fig1():
    fig, ax = plt.subplots(figsize=(6.8, 2.7)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.01, 0.62, 0.17, 0.26, "Non-replayable inputs\n(LLM proposals,\nhuman judgment)", fc=GREYS[3])
    box(ax, 0.23, 0.62, 0.17, 0.26, "Frozen registry\n(design space, signs,\nsearch policy)\nSHA-256 hash")
    box(ax, 0.45, 0.62, 0.19, 0.26, "Replayable loop L\n(code only: search,\nstop, report)", bold=True)
    box(ax, 0.70, 0.62, 0.28, 0.26, "Real data D  →  report R\n(package, statistic T)")
    box(ax, 0.23, 0.10, 0.22, 0.30, "Null worlds D(b)\npermute treatment /\nmoderator within strata\n(effects absent)", fc=GREYS[3])
    box(ax, 0.50, 0.10, 0.22, 0.30, "Replay L on every D(b)\n→ Potemkin rate\n→ search-adjusted p", bold=True)
    box(ax, 0.77, 0.10, 0.21, 0.30, "Planted worlds\n(known effects)\n→ power, winner's curse")
    arrow(ax, 0.18, 0.75, 0.23, 0.75); arrow(ax, 0.40, 0.75, 0.45, 0.75); arrow(ax, 0.64, 0.75, 0.70, 0.75)
    arrow(ax, 0.34, 0.62, 0.34, 0.40, ls="--"); arrow(ax, 0.45, 0.25, 0.50, 0.25); arrow(ax, 0.72, 0.25, 0.77, 0.25)
    arrow(ax, 0.61, 0.40, 0.80, 0.62, ls=":")
    ax.text(0.745, 0.47, "rank real T among\nnull reports", fontsize=6.8, ha="left", va="center")
    fig.savefig(F / "fig1_workflow.pdf"); plt.close(fig)


def fig2(nwc):
    r = nwc.get("potemkin_rates", {})
    if not r:
        return
    names = list(r); vals = [100 * r[n]["rate"] for n in names]
    lo = [100 * (r[n]["rate"] - r[n]["lo"]) for n in names]; hi = [100 * (r[n]["hi"] - r[n]["rate"]) for n in names]
    if "guard_null" in nwc:
        g = nwc["guard_null"]; names.append("Guard (split + Holm on held-out half)")
        vals.append(100 * g["package_rate"]); lo.append(100 * (g["package_rate"] - g["lo"])); hi.append(100 * (g["hi"] - g["package_rate"]))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    y = np.arange(len(names))[::-1]
    cols = [GREYS[0], GREYS[1], GREYS[2], GREYS[2], GREYS[3], GREYS[3]][: len(names)]
    ax.barh(y, vals, xerr=[lo, hi], color=cols, edgecolor=GREYS[0], lw=0.6, error_kw=dict(lw=0.7, capsize=2))
    ax.axvline(5, color=GREYS[0], ls="--", lw=0.8); ax.text(5.5, y.max() + 0.45, "nominal 5%", fontsize=7)
    for yi, v, h in zip(y, vals, hi):
        ax.text(v + h + 1.5, yi, f"{v:.1f}%", va="center", fontsize=7)
    names = [n.replace("NWC (loop-level p <= .05)", "NWC (exact by construction)").replace(">=", "\u2265") for n in names]
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Null worlds reporting a fully significant H1–H4 package (%)")
    fig.savefig(F / "fig2_potemkin.pdf"); plt.close(fig)


def fig3(tb, nwc):
    es = pd.DataFrame(tb["event_study"])
    null = pd.read_csv(ROOT / "intermediate/14_null_summary.csv") if (ROOT / "intermediate/14_null_summary.csv").exists() else None
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.3))
    a = ax[0]
    a.errorbar(es.year, 100 * es.b, yerr=196 * es.se, fmt="o", color=GREYS[0], ms=3, capsize=2, lw=0.8)
    a.axhline(0, color=GREYS[2], lw=0.6); a.axvline(2022.5, color=GREYS[2], ls=":", lw=0.8)
    a.set_xlabel("Year"); a.set_ylabel("Effect of 1-SD exposure (pp)"); a.set_title("(a) Event study", loc="left")
    a.set_xticks([2016, 2018, 2020, 2022, 2024]); a.set_xticklabels(["2016", "2018", "2020", "2022", "2024"])
    if null is not None:
        for i, (col, real, lab) in enumerate([("h1_max", nwc["real"]["h1_max"], "(b) H1: max t over search"),
                                               ("U_T", nwc["real"]["U_T"], "(c) Package: min t, loop U")]):
            a = ax[1 + i]
            v = null[col].replace(-np.inf, np.nan).dropna()
            a.hist(v, bins=30, color=GREYS[3], edgecolor=GREYS[2], lw=0.4)
            a.axvline(real, color=GREYS[0], lw=1.2); a.axvline(1.96, color=GREYS[1], ls="--", lw=0.8)
            p = nwc["p_nwc_h1_max"] if col == "h1_max" else nwc["p_nwc_package_U"]
            a.text(real, a.get_ylim()[1] * 0.92, f" real\n p={p:.3f}", fontsize=7, va="top")
            a.set_title(lab, loc="left"); a.set_xlabel("Statistic in null worlds"); a.set_ylabel("Worlds")
    fig.tight_layout(); fig.savefig(F / "fig3_testbed.pdf"); plt.close(fig)


def fig4(pkg_mods, signs, labels):
    """Conceptual diagram: moderators point at the H1 relationship (arrow), not at the boxes."""
    fig, ax = plt.subplots(figsize=(6.4, 3.1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.01, 0.28, 0.25, 0.22, "Pre-shock workforce exposure\nto large language models\n\u00d7 post-ChatGPT (2023\u20132025)", fs=7.2)
    box(ax, 0.74, 0.28, 0.25, 0.22, "Experience tilt of hiring\n(fewer entry-level hires,\nmore experienced hires)", fs=7.2)
    ax.add_patch(FancyArrowPatch((0.265, 0.39), (0.735, 0.39), arrowstyle="-|>", mutation_scale=10, lw=1.6, color=GREYS[0]))
    ax.text(0.50, 0.33, "H1 (+)", ha="center", va="top", fontsize=8, weight="bold")
    xs = [0.335, 0.50, 0.665]
    for i, (m, s_, lab) in enumerate(zip(pkg_mods, signs, labels)):
        x = xs[i]
        box(ax, x - 0.065, 0.70, 0.13, 0.19, lab, fc=GREYS[3] if s_ < 0 else "#FFFFFF", fs=6.4)
        ax.add_patch(FancyArrowPatch((x, 0.695), (x, 0.405), arrowstyle="-|>", mutation_scale=8, lw=1.0,
                                     ls="-" if s_ > 0 else "--", color=GREYS[0]))
        sg = "+" if s_ > 0 else "\u2212"
        ax.text(x + 0.012, 0.55, f"H{i+2} ({sg})", fontsize=7.2, ha="left", va="center")
    ax.text(0.5, 0.08, "Solid arrow: the condition strengthens H1 (amplifies); dashed arrow, shaded box: it weakens H1 (attenuates).\n"
            "All moderators measured before the shock (2016\u20132021). Firm and industry-by-year fixed effects; SE clustered by firm.",
            ha="center", fontsize=6.6)
    fig.savefig(F / "fig4_conceptual.pdf"); plt.close(fig)


def fig5(nwc):
    pl = nwc.get("planted", {})
    if not pl:
        return
    d = pd.DataFrame(pl.values()).sort_values("beta_star")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for i, (col, lab) in enumerate([("U_power", "Loop U (nominal)"), ("Ufree_power", "Loop U-free (nominal)"),
                                    ("nwc_power", "NWC (search-adjusted)"), ("bonf_power", "Bonferroni")]):
        if col in d and d[col].notna().any():
            ax[0].plot(d.beta_star, 100 * d[col], LINES[i % 4], color=GREYS[min(i, 3)], marker=MARKERS[i % 4], ms=4, label=lab)
    ax[0].set_xlabel("Planted main effect (within-firm SD per SD of exposure)"); ax[0].set_ylabel("Worlds reporting the package (%)")
    ax[0].legend(frameon=False, fontsize=6.8)
    if "winner_ratio_median" in d and d.winner_ratio_median.notna().any():
        ax[1].plot(d.beta_star, d.winner_ratio_median, "-", color=GREYS[0], marker="o", ms=4)
        ax[1].axhline(1, color=GREYS[2], ls="--", lw=0.8)
        ax[1].set_xlabel("Planted main effect"); ax[1].set_ylabel("Reported / planted main effect (median)")
    fig.tight_layout(); fig.savefig(F / "fig5_planted.pdf"); plt.close(fig)


def fig_spec():
    sc = pd.read_csv(ROOT / "intermediate/10_spec_curve.csv")
    sc = sc.sort_values("t_h1").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    mk = {"tilt_exp": "o", "hire_sen_mean": "s", "entry_rate": "^"}
    for i, o in enumerate(["tilt_exp", "hire_sen_mean", "entry_rate"]):
        s = sc[sc.outcome == o]
        ax.scatter(s.index, s.t_h1, s=4, marker=mk[o], color=GREYS[i], label=o.replace("_", " "))
    ax.axhline(1.96, color=GREYS[0], ls="--", lw=0.8); ax.axhline(0, color=GREYS[2], lw=0.5)
    ax.set_xlabel("Registered base specifications, ranked by t"); ax.set_ylabel("Sign-consistent t for H1")
    ax.legend(frameon=False, markerscale=2)
    fig.savefig(F / "figA_spec_curve.pdf"); plt.close(fig)


def main():
    setup()
    tb = json.loads((ROOT / "tables/09_testbed.json").read_text())
    nwc = json.loads((ROOT / "tables/nwc.json").read_text()) if (ROOT / "tables/nwc.json").exists() else {}
    fig1(); fig_spec()
    if nwc:
        fig2(nwc); fig3(tb, nwc); fig5(nwc)
    real = json.loads((ROOT / "intermediate/10_loop_real.json").read_text())
    pkg = real["U"]["first_success"]
    import yaml
    reg = yaml.safe_load((ROOT / "registry/design_space.yaml").read_text())
    info = {m["name"] + "_z": m for m in reg["moderators"]}
    mods = pkg["moderators"]
    signs = [1 if info[m]["sign"] == "amplifies" else -1 for m in mods]
    order = sorted(range(3), key=lambda i: -signs[i])
    labs = {"pyramid_z": "Bottom-heavy\nhierarchy", "remote_z": "Digitizable\nwork",
            "firm_age_z": "Organizational\nage", "rd_int_z": "Knowledge intensity\n(R&D / assets)",
            "analysts_z": "Analyst coverage", "slack_cash_z": "Financial slack", "ilm_share_z": "Internal labor market",
            "ai_talent_pre_z": "Prior AI talent", "entry_wage_z": "Junior labor cost", "hhi_ind_z": "Industry concentration"}
    fig4([mods[i] for i in order], [signs[i] for i in order], [labs[mods[i]] for i in order])
    print("figures written")


if __name__ == "__main__":
    main()
