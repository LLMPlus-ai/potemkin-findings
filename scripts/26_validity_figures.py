"""26 - Figures for the validity analyses (plan_10_20.tex, technical appendix; four greys, greyscale-legible).

Fig 7  fig7_exchangeability.pdf  rejection rate vs the spurious effect a non-exchangeable trait induces
Fig 8  fig8_moderator_worlds.pdf Monte Carlo size of moderator tests by construction; real-data p-values
Fig 9  fig9_selection.pdf        what the moderator p-value measures (selection vs strength) and the
                                 familywise correction over all ten registered candidates
Fig 10 fig10_budget.pdf          Potemkin rate as a function of the search budget (models visited)
Input: tables/validity.json (scripts/25)
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from plotstyle import setup, GREYS, plt
from matplotlib.lines import Line2D

F = ROOT / "figures"
MODS3 = ["pyramid_z", "remote_z", "firm_age_z"]
ROWLAB = {"pyramid_z": "H2 hierarchy\nshape", "remote_z": "H3 digitizable\nwork", "firm_age_z": "H4 organizational\nage"}


def style():
    setup()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.titlepad": 4})


def five(ax, horizontal=False):
    (ax.axvline if horizontal else ax.axhline)(5, color=GREYS[0], ls="--", lw=0.7, zorder=1)


def fig_exchangeability(v):
    ex = pd.DataFrame(v["mc_exch"]["rows"])
    post = ex[(ex.channel == "post")].sort_values("spurious_pct")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.55))
    series = [("single", "One pre-specified regression", "--", "s", GREYS[1], GREYS[1]),
              ("naive", "Naive loop (any spec. t ≥ 1.96)", ":", "^", GREYS[1], "#FFFFFF"),
              ("NWC", "NWC, within-sector worlds", "-", "o", GREYS[0], GREYS[0]),
              ("NWC_cov", "NWC, trait-preserving worlds", "-", "D", GREYS[2], "#FFFFFF")]
    for k, lab, ls, mk, col, mfc in series:
        a.plot(post.spurious_pct, 100 * post[k], ls=ls, marker=mk, ms=3.6, color=col, mfc=mfc, mec=col, lw=0.9, label=lab)
    five(a)
    a.set_xlabel("Spurious post-shock effect induced by the trait (% of the real H1 estimate)")
    a.set_ylabel("Rejection rate in worlds with no effect (%)")
    a.set_ylim(-3, 103); a.set_xlim(left=0)
    a.legend(frameon=False, loc="center right", handlelength=2.2)
    a.set_title("a  Exposure proxies for a trait that shifts outcomes")
    a.text(0.02, 0.97, "dashed line: 5%", transform=a.transAxes, fontsize=6.3, va="top", color=GREYS[1])
    # b: the cases where exchangeability holds, or the trait is observed
    rows = [("null", "No trait (null)"), ("level", "Trait shifts levels only\n(ρ = 0.6)"),
            ("post_r0.6_d0.1", "Post-shock shift,\nρ = 0.6, δ = 0.10"), ("post_r0.6_d0.2", "Post-shock shift,\nρ = 0.6, δ = 0.20")]
    e = ex.set_index("cell")
    y = np.arange(len(rows))[::-1]
    marks = [("NWC", "o", GREYS[0], GREYS[0], "within-sector worlds"), ("NWC_strat", "s", GREYS[0], "#FFFFFF", "strata on trait quintile"),
             ("NWC_cov", "D", GREYS[2], "#FFFFFF", "trait-preserving worlds"), ("single", "|", GREYS[1], GREYS[1], "one pre-specified regression")]
    off = np.linspace(-0.24, 0.24, len(marks))
    for (k, mk, col, mfc, lab), dy in zip(marks, off):
        xs = [100 * e.loc[c, k] if c in e.index else np.nan for c, _ in rows]
        b.plot(xs, y + dy, ls="none", marker=mk, ms=4 if mk != "|" else 7, color=col, mfc=mfc, mec=col, label=lab, mew=0.9)
    five(b, horizontal=True)
    b.set_yticks(y); b.set_yticklabels([r for _, r in rows]); b.set_xlim(-3, 103)
    b.set_xlabel("Rejection rate (%)")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handletextpad=0.3, columnspacing=0.8,
             borderaxespad=0.1)
    b.set_title("b  Remedies when the trait is observed")
    fig.savefig(F / "fig7_exchangeability.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_moderators(v):
    mm = pd.DataFrame(v["mc_mod"]).set_index("scenario")
    lab = {"clean": "No other structure", "trend": "Own post-shock trend", "proxy": "Correlated moderator\nwith a true interaction",
           "nonlinear": "Concave exposure effect", "all": "All three"}
    order = [s for s in ["clean", "trend", "proxy", "nonlinear", "all"] if s in mm.index]
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.25))
    y = np.arange(len(order))[::-1]
    marks = [("nominal", "|", GREYS[1], GREYS[1], "nominal t ≥ 1.96"), ("perm", "o", GREYS[0], "#FFFFFF", "permute (Type B)"),
             ("residperm", "s", GREYS[2], GREYS[2], "conditional permute"), ("FL", "^", GREYS[0], GREYS[0], "null model (Freedman–Lane)")]
    off = np.linspace(-0.24, 0.24, len(marks))
    for ax, stat, ttl in [(axs[0], "free", "a  With search (screening t)"), (axs[1], "fix", "b  One specification")]:
        for (k, mk, col, mfc, lb), dy in zip(marks, off):
            xs = [100 * mm.loc[s, f"{k}_{stat}"] for s in order]
            ax.plot(xs, y + dy, ls="none", marker=mk, ms=4 if mk != "|" else 7, color=col, mfc=mfc, mec=col, mew=0.9, label=lb)
        five(ax, horizontal=True)
        ax.set_xscale("symlog", linthresh=10, linscale=1.2); ax.set_xlim(0, 100)
        ax.set_xticks([0, 5, 10, 50, 100]); ax.set_xticklabels(["0", "5", "10", "50", "100"])
        ax.set_yticks(y); ax.set_yticklabels([lab[s] for s in order] if stat == "free" else [])
        ax.set_xlabel("Rejection rate (%), true interaction = 0"); ax.set_title(ttl)
    axs[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=4, handletextpad=0.3, columnspacing=1.2)
    # c: real data, the reported-package statistic under the three constructions
    c, d = v["constructions"], v["decomposition"]
    ax = axs[2]
    yy = np.arange(3)[::-1]
    cm = [("permB", "o", GREYS[0], "#FFFFFF"), ("residB", "s", GREYS[2], GREYS[2]), ("fl", "^", GREYS[0], GREYS[0])]
    off = [-0.2, 0, 0.2]
    for (cc, mk, col, mfc), dy in zip(cm, off):
        if not all(c.get(cc, {}).get(m, {}).get("B", 0) >= 500 for m in MODS3):
            continue
        ax.plot([c[cc][m]["p_sel"] for m in MODS3], yy + dy, ls="none", marker=mk, ms=4.2, color=col, mfc=mfc, mec=col, mew=0.9)
    ax.axvline(0.05, color=GREYS[0], ls="--", lw=0.7)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1.5); ax.set_xticks([1e-3, 1e-2, 0.05, 1]); ax.set_xticklabels(["0.001", "0.01", "0.05", "1"])
    ax.set_yticks(yy); ax.set_yticklabels([ROWLAB[m] for m in MODS3]); ax.yaxis.tick_right()
    ax.set_xlabel("Search-adjusted p, reported package"); ax.set_title("c  Real data: H2–H4")
    fig.savefig(F / "fig8_moderator_worlds.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_selection(v):
    d, f, c = v["decomposition"], v.get("familywise"), v["constructions"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True, gridspec_kw=dict(width_ratios=[0.8, 1.25], wspace=0.08))
    y = np.arange(3)[::-1].astype(float)
    for ax in (a, b):
        for yi in (0.5, 1.5):
            ax.axhline(yi, color=GREYS[3], lw=0.5, zorder=0)
    # a: how often the loop reports the permuted moderator
    for m, yi in zip(MODS3, y):
        a.barh(yi, 100 * d[m]["share_selected"], height=0.42, color=GREYS[3], edgecolor=GREYS[2], lw=0.4)
        a.text(100 * d[m]["share_selected"] + 2.5, yi, f"{100 * d[m]['share_selected']:.1f}%", va="center", fontsize=6.6)
    a.set_xlim(0, 118); a.set_yticks(y); a.set_yticklabels([ROWLAB[m] for m in MODS3]); a.set_ylim(-0.5, 2.5)
    a.set_xlabel("Permuted worlds in which the loop\nreports the permuted moderator (%)")
    a.set_title("a  How often the loop selects it")
    # b: p-values by statistic, and the familywise adjustments over all ten candidates
    marks = [("p_sel", "o", GREYS[0], GREYS[0], "t in reported package (selection + strength)"),
             ("p_free", "s", GREYS[0], "#FFFFFF", "screening t, defined in every world"),
             ("p_fix", "D", GREYS[2], "#FFFFFF", "reported specification, no search")]
    off = [0.28, 0.14, 0.0]
    for (k, mk, col, mfc, lb), dy in zip(marks, off):
        b.plot([d[m][k] for m in MODS3], y + dy, ls="none", marker=mk, ms=4.0, color=col, mfc=mfc, mec=col, mew=0.9, label=lb)
    b.plot([c["permB"][m]["holm10"] for m in MODS3], y - 0.14, ls="none", marker="o", ms=4.0, color=GREYS[0], mfc="#FFFFFF",
           mew=0.9, label="familywise: Holm over all ten (primary)")
    if f:
        b.plot([f["minP_sel"][m]["stepdown"] for m in MODS3], y - 0.28, ls="none", marker="x", ms=4.2, color=GREYS[1], mew=0.9,
               label="familywise: joint worlds, step-down min-P")
    b.axvline(0.05, color=GREYS[0], ls="--", lw=0.7)
    b.set_xscale("log"); b.set_xlim(1e-3, 1.6); b.set_xticks([1e-3, 1e-2, 0.05, 1]); b.set_xticklabels(["0.001", "0.01", "0.05", "1"])
    b.tick_params(axis="y", length=0)
    b.set_xlabel("Search-adjusted p in the moderator's permuted worlds (log scale)")
    b.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), handletextpad=0.3, fontsize=6.4)
    b.set_title("b  What the p-value measures")
    fig.savefig(F / "fig9_selection.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_budget(v):
    bu = v["budget"]
    g = np.array(bu["grid"])
    U, Fr = 100 * np.array(bu["U"]), 100 * np.array(bu["Ufree"])
    fig, ax = plt.subplots(figsize=(4.4, 2.75))
    ax.step(g, U, where="post", color=GREYS[0], lw=1.1)
    ax.step(g, Fr, where="post", color=GREYS[0], lw=1.1, ls="--")
    at = lambda arr, n: arr[np.searchsorted(g, n, side="right") - 1]
    nU, nF = bu["real_U"], bu["real_Ufree"]
    for n, yv in [(nU, at(U, nU)), (nF, at(Fr, nF))]:
        ax.plot([n], [yv], marker="o", ms=4.5, color=GREYS[0], mfc="#FFFFFF", zorder=5)
    ax.annotate(f"U stopped here on the real data:\n{nU:,} models; {at(U, nU):.1f}% of null worlds\nhave a package by then",
                (nU, at(U, nU)), xytext=(60, 60), ha="center", fontsize=6.2, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GREYS[1], shrinkB=3))
    ax.annotate(f"U-free stopped here:\n{nF} models; {at(Fr, nF):.1f}%", (nF, at(Fr, nF)), xytext=(2.2, 30), ha="center",
                fontsize=6.2, linespacing=1.1, arrowprops=dict(arrowstyle="-", lw=0.5, color=GREYS[1], shrinkB=3))
    ax.axhline(5, color=GREYS[2], ls=":", lw=0.8)
    ax.text(70, 6.3, "NWC: 5% by construction", fontsize=6.2, color=GREYS[1])
    ax.legend(handles=[Line2D([], [], color=GREYS[0], lw=1.1, label="U (registered signs)"),
                       Line2D([], [], color=GREYS[0], lw=1.1, ls="--", label="U-free (signs rationalized)")],
              frameon=False, loc="upper left", handlelength=2.4)
    ax.set_xscale("log"); ax.set_xlim(1, g[-1]); ax.set_ylim(0, 100)
    ax.set_xlabel("Search budget: models the loop may fit before it must report")
    ax.set_ylabel("Null worlds with a full package (%)")
    fig.savefig(F / "fig10_budget.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    style()
    v = json.loads((ROOT / "tables/validity.json").read_text())
    if v.get("mc_exch", {}).get("rows"):
        fig_exchangeability(v)
    if v.get("mc_mod"):
        fig_moderators(v)
    fig_selection(v)
    fig_budget(v)
    print("figures written")


if __name__ == "__main__":
    main()
