"""20 - Aggregate Monte Carlo results on synthetic twins (script 13).

Rejection rates by scenario for: the unguarded loop U (reports a fully significant
package), Bonferroni over models visited, the specification-curve rule, and
Null-World Calibration (loop-level p <= 0.05 for the package and for H1).
Outputs: tables/mc.json (macros read by script 17), tables/tab_mc.tex
"""
import sys, json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from nullworld.infer import mc_band

LAB = {"null": "Null (no effect; treatment exchangeable within sector)",
       "leakage": "Null, moderators chosen by peeking at outcomes (not replayed)",
       "leakage_replayed": "Null, the same peeking step replayed in every null world",
       "confounded": "No causal effect; unobserved trait drives both exposure and post-period drift",
       "power": "Planted effects (H1 0.10 SD; three moderators $\\pm$0.06 SD)"}


def main():
    rows, out = [], {}
    for scen in ["null", "leakage", "leakage_replayed", "confounded", "power"]:
        import re
        fs = [f for f in glob.glob(str(ROOT / f"intermediate/13_mc/{scen}_*.json"))
              if re.fullmatch(rf"{scen}_\d+\.json", Path(f).name)]
        if not fs:
            continue
        d = pd.DataFrame([json.loads(Path(f).read_text()) for f in fs])
        R, B = len(d), int(d.B.iloc[0])
        r = dict(scenario=scen, R=R, B=B,
                 U=float(d.U_success.mean()), BONF=float(d.BONF_success.mean()), SPEC=float(d.SPEC_success.mean()),
                 NWC_pkg=float((d.p_nwc_package <= 0.05).mean()), NWC_h1=float((d.p_nwc_h1 <= 0.05).mean()))
        rows.append(r)
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(ROOT / "intermediate/20_mc_summary.csv", index=False)
        g = t.set_index("scenario")
        f = lambda s, c: f"{100 * g.loc[s, c]:.1f}" if s in g.index else "\\textbf{[pending]}"
        out = {"sizeNWC": f("null", "NWC_h1"), "sizeNWCpkg": f("null", "NWC_pkg"), "sizeU": f("null", "U"),
               "sizeSpec": f("null", "SPEC"), "sizeLeak": f("leakage", "NWC_h1"),
               "sizeLeakReplay": f("leakage_replayed", "NWC_h1"), "sizeConf": f("confounded", "NWC_h1"),
               "powerNWChOne": f("power", "NWC_h1"),
               "powerNWC": f("power", "NWC_pkg"), "powerU": f("power", "U"),
               "R": str(int(g.R.max())), "B": str(int(g.B.max()))}
        (ROOT / "tables/mc.json").write_text(json.dumps(out, indent=1))
        L = ["\\begin{table}[!htbp]\\centering\\footnotesize", "\\caption{Monte Carlo on synthetic twins: rejection rates (\\%)}\\label{tab:mc}",
             "\\begin{tabular}{p{6.2cm}rrrrrr}", "\\toprule",
             "Scenario & $R$ & U & Bonferroni & Spec.\\ curve & NWC package & NWC H1 \\\\", "\\midrule"]
        for r in rows:
            lo, hi = mc_band(r["NWC_pkg"], r["R"])
            L.append(f"{LAB[r['scenario']]} & {r['R']} & {100*r['U']:.1f} & {100*r['BONF']:.1f} & {100*r['SPEC']:.1f} & "
                     f"{100*r['NWC_pkg']:.1f} & {100*r['NWC_h1']:.1f} \\\\")
        L += ["\\bottomrule", "\\end{tabular}",
              "\\begin{minipage}{0.95\\linewidth}\\vspace{2pt}\\scriptsize\\textit{Notes.} Each replication draws a synthetic twin "
              "(Section~\\ref{sec:mc}), runs the reduced registered loop (12 base specifications and the 10-moderator search), and "
              f"computes loop-level $p$-values from $B$ inner null worlds (within-sector permutation of exposure). Under the null, "
              "NWC rejection rates should not exceed 5\\% up to Monte Carlo error; Theorem 1 does not cover the leakage and confounded "
              "scenarios, which show what the method cannot do.\\end{minipage}", "\\end{table}"]
        (ROOT / "tables/tab_mc.tex").write_text("\n".join(L))
        print(t.to_string())


if __name__ == "__main__":
    main()
