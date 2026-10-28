"""12 - Descriptive evidence ("show the data alive", OS transparency policy):
kernel densities, raw event-time means by exposure tercile, the generative-AI
adoption curve by exposure tercile, a bin scatter of the pre/post change
against exposure, a balance table by exposure tercile, and summary
statistics / correlations of all focal variables.
Outputs: figures/desc_*.pdf, tables/desc_*.csv, tables/desc.json
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from plotstyle import setup, GREYS, LINES, MARKERS, plt
from scipy import stats

MODS = ["ilm_share", "slack_cash", "analysts", "firm_age", "remote", "ai_talent_pre", "pyramid", "entry_wage", "hhi_ind", "rd_int"]
CTRL = ["log_hc_l1", "hire_rate_l1", "sep_rate_l1", "log_at_l1", "roa_l1", "lev_l1", "sales_growth_l1", "rd_at_l1"]


def main():
    setup()
    p = pd.read_parquet(ROOT / "data/analysis/panel.parquet")
    f = pd.read_parquet(ROOT / "data/analysis/firms.parquet")
    f["terc"] = pd.qcut(f.expo_beta, 3, labels=["Low", "Middle", "High"])
    p = p.merge(f[["gvkey", "terc", "expo_beta"]], on="gvkey")
    out = {}
    # raw event-time means of the outcome and of adoption by exposure tercile
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for i, t in enumerate(["Low", "Middle", "High"]):
        s = p[p.terc == t].groupby("year")
        ax[0].plot(s.tilt_exp.mean().index, 100 * s.tilt_exp.mean(), LINES[i], color=GREYS[i], marker=MARKERS[i], ms=3, label=f"{t} exposure")
        ax[1].plot(s.genai_share.mean().index, s.genai_share.mean(), LINES[i], color=GREYS[i], marker=MARKERS[i], ms=3, label=f"{t} exposure")
    for a in ax:
        a.axvline(2022.5, color=GREYS[2], lw=0.8, ls=":")
    ax[0].set_ylabel("Hires at associate level or above (%)"); ax[0].set_xlabel("Year")
    ax[1].set_ylabel("Workforce with dated GenAI skills (%)"); ax[1].set_xlabel("Year")
    ax[0].legend(frameon=False)
    fig.savefig(ROOT / "figures/desc_event_time_means.pdf"); plt.close(fig)
    et = p.groupby(["terc", "year"])[["tilt_exp", "genai_share", "hire_sen_mean", "entry_rate"]].mean().reset_index()
    et.to_csv(ROOT / "tables/desc_event_time_means.csv", index=False)

    # bin scatter of the post-minus-pre change against exposure
    ch = (p[p.post == 1].groupby("gvkey").tilt_exp.mean() - p[p.post == 0].groupby("gvkey").tilt_exp.mean()).dropna()
    x = f.set_index("gvkey").expo_beta.reindex(ch.index)
    b = pd.qcut(x, 20, labels=False)
    bs = pd.DataFrame({"x": x.groupby(b).mean(), "y": 100 * ch.groupby(b).mean()})
    sl = stats.linregress(x, 100 * ch)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.scatter(bs.x, bs.y, s=12, color=GREYS[0], marker="o")
    xx = np.linspace(x.min(), x.max(), 50); ax.plot(xx, sl.intercept + sl.slope * xx, "--", color=GREYS[1])
    ax.set_xlabel("Pre-shock LLM exposure (GPT-4 beta)"); ax.set_ylabel("Change in experienced-hire share (pp)")
    fig.savefig(ROOT / "figures/desc_binscatter_change.pdf"); plt.close(fig)
    out["binscatter_slope_pp_per_unit"] = float(sl.slope); out["binscatter_p"] = float(sl.pvalue)
    out["binscatter_n_firms"] = int(len(ch))

    # kernel densities of exposure and outcome
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
    kd = stats.gaussian_kde(f.expo_beta.dropna()); xs = np.linspace(f.expo_beta.min(), f.expo_beta.max(), 200)
    ax[0].plot(xs, kd(xs), color=GREYS[0]); ax[0].set_xlabel("Pre-shock LLM exposure"); ax[0].set_ylabel("Density")
    for i, (lab, d) in enumerate([("2016-2022", p[p.post == 0]), ("2023-2025", p[p.post == 1])]):
        k2 = stats.gaussian_kde(100 * d.tilt_exp); xs2 = np.linspace(0, 100, 200)
        ax[1].plot(xs2, k2(xs2), LINES[i], color=GREYS[i], label=lab)
    ax[1].set_xlabel("Hires at associate level or above (%)"); ax[1].legend(frameon=False)
    fig.savefig(ROOT / "figures/desc_densities.pdf"); plt.close(fig)

    # balance by exposure tercile (pre-period firm characteristics)
    pre = p[p.year.between(2019, 2021)].groupby("gvkey")[["tilt_exp", "hire_sen_mean", "entry_rate"] + CTRL].mean()
    fz = f.set_index("gvkey")
    bal = pre.join(fz[MODS + ["terc"]])
    rows = []
    for v in ["tilt_exp", "hire_sen_mean", "entry_rate"] + CTRL + MODS:
        lo, hi = bal.loc[bal.terc == "Low", v], bal.loc[bal.terc == "High", v]
        nd = (hi.mean() - lo.mean()) / np.sqrt((hi.var() + lo.var()) / 2)
        rows.append(dict(variable=v, low=lo.mean(), middle=bal.loc[bal.terc == "Middle", v].mean(), high=hi.mean(), norm_diff=nd))
    pd.DataFrame(rows).to_csv(ROOT / "tables/desc_balance.csv", index=False)

    # summary statistics and correlations
    zz = p.merge(f[["gvkey"] + MODS], on="gvkey", suffixes=("", "_f"))
    cols = ["tilt_exp", "hire_sen_mean", "entry_rate", "expo_beta", "genai_share"] + MODS + CTRL
    ss = zz[cols].describe(percentiles=[0.25, 0.5, 0.75]).T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    ss.to_csv(ROOT / "tables/desc_summary.csv")
    zz[cols].corr().to_csv(ROOT / "tables/desc_correlations.csv")
    out["n_firm_years"] = int(len(p)); out["n_firms"] = int(p.gvkey.nunique())
    (ROOT / "tables/desc.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)); print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
