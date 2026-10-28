"""28 - Independent check of the validity numbers cited in the proposal and the technical appendix.

Recomputes each key number from the raw world files with its own counting code (no import of nullworld.infer or
scripts/25) and compares it with the macro in manuscript/numbers_validity.tex. Also checks the revised texts for
the phrases the review asked to remove and for unresolved placeholders. Prints PASS/FAIL per check and writes
intermediate/28_check.json.
usage: python3 scripts/28_check_validity_numbers.py [proposal.tex ...]
"""
import sys, json, glob, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, pandas as pd

MODS3 = ["pyramid_z", "remote_z", "firm_age_z"]
TAG = {"pyramid_z": "Pyr", "remote_z": "Rem", "firm_age_z": "Age"}


def J(pat):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / pat)))]


def pv(obs, null):
    null = [(-1e300 if v is None else v) for v in null]
    return (1 + sum(1 for v in null if v >= obs)) / (1 + len(null))


def macros():
    m = {}
    for line in (ROOT / "manuscript/numbers_validity.tex").read_text().splitlines():
        r = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}", line)
        if r:
            m[r.group(1)] = r.group(2)
    for line in (ROOT / "manuscript/numbers.tex").read_text().splitlines():
        r = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}", line)
        if r:
            m[r.group(1)] = r.group(2)
    return m


def main():
    M, res = macros(), []
    def check(name, macro, value, fmt):
        got = M.get(macro, "<missing>")
        want = fmt(value)
        res.append(dict(check=name, macro=macro, macro_value=got, recomputed=want, ok=(got == want)))
    f3 = lambda x: f"{x:.3f}"
    f1 = lambda x: f"{x:.1f}"
    real = json.loads((ROOT / "intermediate/10_loop_real.json").read_text())
    pkg = real["U"]["first_success"]
    obs = json.loads((ROOT / "intermediate/22_real.json").read_text())
    tobs = {m: pkg["t_mods"][pkg["moderators"].index(m)] for m in MODS3}
    # one-at-a-time permutation worlds: selection share, p_sel (raw scripts/11 files), p_free, p_fix
    psel = {}
    for m in MODS3:
        W11 = J(f"intermediate/11_typeB/B_{m}_*.json")
        sel = [w["U"]["pkg"]["t_mods"][w["U"]["pkg"]["moderators"].index(m)] if (w["U"]["success"] and m in w["U"]["pkg"]["moderators"]) else None for w in W11]
        psel[m] = pv(tobs[m], sel)
        check(f"share selected {m}", f"VSel{TAG[m]}", 100 * np.mean([s is not None for s in sel]), f1)
        check(f"p_sel {m}", f"VpSel{TAG[m]}", psel[m], f3)
        W = J(f"intermediate/22_permB/permB_{m}_*.json")
        check(f"p_free {m}", f"VpFree{TAG[m]}", pv(obs["free"][m], [w["free"] for w in W]), f3)
        check(f"p_fix {m}", f"VpFix{TAG[m]}", pv(tobs[m], [w["fix"] for w in W]), f3)
        for c, ct in [("fl", "FL"), ("residB", "Res")]:
            W = J(f"intermediate/22_{c}/{c}_{m}_*.json")
            if len(W) >= 500:
                check(f"{c} p_sel {m}", f"Vp{ct}Sel{TAG[m]}", pv(tobs[m], [w["sel"] for w in W]), f3)
                check(f"{c} p_free {m}", f"Vp{ct}Free{TAG[m]}", pv(obs["free"][m], [w["free"] for w in W]), f3)
    # Holm over ten by hand: sorted p times (10, 9, 8, ...), running max, capped at 1
    ps = sorted([(psel[m], m) for m in MODS3] + [(1.0, f"u{i}") for i in range(7)])
    run, holm = 0.0, {}
    for i, (p, m) in enumerate(ps):
        run = max(run, min(1.0, (10 - i) * p)); holm[m] = run
    for m in MODS3:
        check(f"Holm-10 {m}", f"VHolmTen{TAG[m]}", holm[m], f3)
    # joint worlds: full-package share; brute-force step-down min-P for the three reported moderators
    JW = J("intermediate/22_joint/joint_*.json")
    if JW:
        check("joint success share", "VJointSuccess", 100 * np.mean([w["success"] for w in JW]), f1)
        allm = list(obs["free"])
        S = {j: np.array([w["sel"].get(j, -np.inf) for w in JW]) for j in allm}
        o = {j: (tobs[j] if j in tobs else -np.inf) for j in allm}
        B = len(JW)
        P0 = {j: (1 + np.sum(S[j] >= o[j])) / (B + 1) for j in allm}
        Pb = {j: np.array([np.sum(np.append(S[j], o[j]) >= v) / (B + 1) for v in S[j]]) for j in allm}
        order = sorted(allm, key=lambda j: P0[j]); run = 0.0; sd = {}
        for i, j in enumerate(order):
            mp = np.min(np.column_stack([Pb[r] for r in order[i:]]), 1)
            run = max(run, (1 + np.sum(mp <= P0[j])) / (B + 1)); sd[j] = min(1.0, run)
        for m in MODS3:
            check(f"joint min-P step-down {m}", f"VMinP{TAG[m]}", sd[m], f3)
    # covariate-preserving worlds
    CA = J("intermediate/22_covA/covA_*.json")
    if CA:
        check("covA p(H1)", "VcovAph", pv(real["U"]["h1_max"], [w["U"]["h1_max"] for w in CA]), f3)
        check("covA p(package)", "VcovApkg", pv(real["U"]["T_first"], [w["U"]["T_first"] for w in CA]), f3)
    # search budget and the random family (raw Type A files)
    A = J("intermediate/11_typeA/A_*.json")
    nU = np.array([w["U"]["n_to_first"] if w["U"]["success"] else np.inf for w in A])
    nF = np.array([w["Ufree"]["n_to_first"] if w["Ufree"]["success"] else np.inf for w in A])
    check("budget U<=1000", "VBudUThousand", 100 * np.mean(nU <= 1000), f1)
    check("budget U<=3483", "VBudUReal", 100 * np.mean(nU <= 3483), f1)
    check("budget Ufree<=12", "VBudFreeTwelve", 100 * np.mean(nF <= 12), f1)
    nm = [w["U"]["n_models"] for w in A]
    check("models min", "VModelsMin", min(nm), lambda x: f"{x:,}")
    check("models max", "VModelsMax", max(nm), lambda x: f"{x:,}")
    pk = [w["U"]["pkg"] for w in A if w["U"]["success"]]
    check("distinct triples", "VTriples", len({tuple(sorted(p["moderators"])) for p in pk}), str)
    # exchangeability Monte Carlo
    for cell, macro in [("null", "VExNull"), ("level", "VExLevel"), ("post_r0.6_d0.1", "VExNWCSixTen"),
                        ("post_r0.6_d0.2", "VExNWCSixTwenty"), ("post_r0.6_d0.2", "VExCovSixTwenty")]:
        W = J(f"intermediate/23_mc/{cell}_[0-9]*.json")
        if W:
            key = "NWC_cov" if macro == "VExCovSixTwenty" else "NWC"
            check(f"MC exch {cell} {key}", macro, 100 * np.mean([w["p"][key] <= 0.05 for w in W]), f1)
    # moderator Monte Carlo
    for s, t in [("all", "All"), ("clean", "Clean")]:
        W = J(f"intermediate/24_mc/{s}_[0-9]*.json")
        if W:
            for k, kt in [("FL_fix", "FLFix"), ("perm_fix", "PermFix")]:
                check(f"MC mod {s} {k}", f"VMm{kt}{t}", 100 * np.mean([w["p"][k] <= 0.05 for w in W]), f1)
    # phrases the review asked to remove; unresolved placeholders
    texts = sys.argv[1:] or ["proposal/stage1_proposal.tex"]
    for tx in texts:
        s = (ROOT / tx).read_text()
        for bad in ["if and only if", "however adaptive", "First procedure-level", "exactly as an AI agent",
                    "Only the link between", "Holm, H2", "27--28 February", "now pure code"]:
            res.append(dict(check=f"{tx}: phrase removed", macro=bad, macro_value=str(bad in s), recomputed="False", ok=bad not in s))
    nump = sum(1 for k, v in M.items() if "pending" in v)
    res.append(dict(check="no pending macros", macro="*", macro_value=str(nump), recomputed="0", ok=nump == 0))
    for r in res:
        print(("PASS" if r["ok"] else "FAIL"), f"{r['check']:<34s} {r['macro']:<22s} macro={r['macro_value']:<10s} recomputed={r['recomputed']}")
    (ROOT / "intermediate/28_check.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"{sum(r['ok'] for r in res)}/{len(res)} checks pass")


if __name__ == "__main__":
    main()
