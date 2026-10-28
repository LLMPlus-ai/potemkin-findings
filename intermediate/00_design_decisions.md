# Design decisions (living record) — OS Special Issue "AI-Enabled Frontiers in Organizational Science"

Public version. Session logistics and notes on drafting the proposal are omitted; every research decision, deviation, and
result is kept as recorded, with its date.

## Contribution type (per CFP)
A methods paper, not a regression paper. It covers four CFP categories: AI research loops under human direction (testing their limits), reusable infrastructure (`nullworld/` package), new approaches to established designs (DiD with moderators), and diagnostic/boundary-setting work.
Anchor: Gartenberg, Hasan, Murray & Pierce (2026, OS) "More versus better". The editorial says AI-guided empirical work "may erode the nuance and judgment that careful empirical work demands", so this progress should be measured rather than assumed.

## Core method: Null-World Calibration (NWC) of AI research loops
- A loop L = deterministic map (data D, frozen design space S, policy pi) -> report R (package + nominal p).
- **Replayability principle:** inference is valid only if every post-data decision is code and can be replayed. Anything that cannot be replayed (LLM or human proposals) must be frozen and SHA-256 hashed before any outcome data are touched.
- Type-A null worlds: permute the firm-level exposure E_i within industry strata. This gives exact randomization inference for the DiD (sharp null) and calibrates H1 and the whole package, yielding the "Potemkin rate".
- Type-B null worlds: permute all moderator candidates within industry x E-quintile strata. Step-down max-T gives search-adjusted p-values for the selected moderators.
- Planted worlds: inject known beta and gamma into Type-A worlds to measure power, recovery, and winner's-curse inflation.
- Theorem to prove in the appendix: the loop-level permutation p is valid for any deterministic, replayable L under within-stratum exchangeability.
- Monte Carlo on synthetic panels calibrated to the real data's moments, to verify size.
- Procedures to compare on size, power, and bias: unguarded loop; Bonferroni; split-sample; specification curve; NWC; guarded protocol (hash + firm split + NWC).

## Empirical testbed (Maryland SMS Level 3, with Level 4 elements)
- Shock: ChatGPT release (2022-11-30). Post period = 2023–2025. Pre period = 2016–2022.
- Unit: firm-year (Compustat NA gvkey linked to Revelio rcid/ultimate parent). Firm FE + year FE, SE clustered by firm; robustness with industry x year FE.
- Exposure E_i = sum_r s_ir(2019–2021 role shares) x E_r. Role exposure E_r comes from a role -> O*NET crosswalk (postings 2018–2022) combined with Eloundou et al. (2024) dv_rating_beta. Alternative proxies: human_rating_beta, Felten LM-AIOE, and leave-one-firm-out role-level GenAI adoption (shift-share).
- H1 (+): E_i x Post raises the experience tilt of hiring (share of hires at seniority >= 3; alternatives: mean hire seniority, and the entry-hire rate with the sign reversed). Theory tension: individual-level experiments show GenAI helps novices most (skill compression), yet organizations may pull up the ladder by substituting junior tasks.
- Level-4 elements: 2SLS with firm GenAI adoption (dated GenAI skills) instrumented by E x Post; shift-share (BHJ/GPSS) argument; event study; placebo shock dates; negative-control exposure (non-LLM AIOE); HonestDiD-style sensitivity; controls for the 2022–23 tech/rate shock (industry x year FE, pre-period characteristics x year).
- Moderator pool (all predetermined, pre-2022; theorized signs must be pre-registered BEFORE any outcome regression): ILM internal-promotion share (-); cash slack (-); analyst coverage (+); firm age (-); remote suitability (+); pre-period AI-core talent share (+); industry HHI (+); hierarchy steepness (+); entry-level salary (+); R&D intensity (sign to be fixed ex ante). H2–H4 = three moderators with mixed signs. No mediator.
- Guard: exploration/confirmation firm split, stratified by industry; confirmation uses one pre-specified model per hypothesis with Holm correction.

## Measurement audits found so far
- Revelio skill_k35000 labels are noisy: 97% of "GitHub Copilot" rows are raw "github"; "Prompt Engineering" absorbs "engineering"; machine translation turns "generative ai" into "generative to". The dictionary is therefore matched on the raw string first, then the translation (scripts/config.py, scripts/01).
- Revelio Positions yearly files are an active-state panel: file Y holds positions active in Y. Hires = rows with year(startdate) = Y. About 43% of startdates are missing.
- The role_k1500 vocabulary equals postings role_k1500_v2 (1,498/1,498).
- rcid -> gvkey: 65,027 linked rcids (15.6k US); 72% are ultimate parents.

## 2026-09-29 04:19:18 — Registry v1 frozen
- registry/design_space.yaml hashed (SHA-256 in registry/design_space.sha256) BEFORE any outcome was computed or any outcome regression run.
- Examined before freezing: data coverage, linkage counts (11,921 linked firms), role exposure distribution (roles only), skill dictionary audit. NOT examined: any hiring-by-seniority outcome, any exposure-outcome association.
- Moderator pool: 10 candidates with theory-fixed signs; ceo_age dropped (ExecuComp covers S&P 1500 only; sign ambiguous between horizon and tech-savviness arguments).

## 2026-09-29 04:20:15 — Implementation decisions not fixed by registry v1 (made before any outcome was seen)
- Analysis sample holds constant across control sets: rows must have non-missing C1 controls (so C0/C1/C2 compare identical rows).
- Missing moderator values are set to the pre-period sample median (standardized value ~0); the share imputed per moderator is reported; complete-case robustness is reported.
- Small-sample correction for cluster-robust SEs: G/(G-1) x (N-1)/(N-K), K excluding absorbed fixed effects (pyfixest default); validated numerically against pyfixest.

## 2026-09-29 04:23:12 — Engine validation and a data-delivery change
- nullworld/ package: fe.py, loop.py, worlds.py, infer.py; engine validated against pyfixest (intermediate/90_engine_validation.csv, max difference ~1e-15).
- 2026-09-29 04:41:13: Positions 2025 file is a new delivery (role_k1500 renamed role_k1500_v2; adds onet_code, msa, city). Handled in script 04. Because a delivery change in the last post year could shift measured seniority, "drop 2025" is a pre-specified robustness check.

## 2026-09-29 04:53:06 — Skill-date audit and the 2SLS endogenous variable (decided before any 2SLS estimate)
- Canonical-label audit (intermediate/01_canonical_label_audit.csv): only 2.6% of rows Revelio labels "GitHub Copilot" are GitHub Copilot (97.4% are raw "github"); "Hugging Face Transformers" 20% valid; "Prompt Engineering" 94%; "Generative AI" 99.7%. Our raw-string dictionary avoids these errors.
- Dictionary hits: 3.92M GenAI rows (2.39M users), 14.7M core-AI rows (7.51M users).
- first_reported dates for GenAI skills: 2023 = 1,779; 2024 = 1.18M; 2025 = 1.74M; 2026 = 0.71M. The date tracks when the resume was captured, not when the skill was acquired. Year-by-year "dated adoption" is therefore a collection artifact.
- Decision: the 2SLS endogenous variable is A_it = (share of the firm's active workforce in year t with ANY GenAI listing) x Post_t, i.e., genai_ever_share x post (zero before 2023 by construction, since GenAI skills cannot predate late 2022). The dated share is reported as a secondary first stage only.
- 2026-09-29 04:53:35 REVISION (still before any 2SLS estimate): the ever-listed GenAI share is 0.70% of the 2016 workforce and 0.96% in 2023 (script 05 audit): it measures workforce COMPOSITION (people who list GenAI skills today), not adoption over time. The 2SLS endogenous variable is therefore genai_share = % of the year-t workforce with a dated GenAI skill first reported by 31 Dec t (0 before 2023; 0.37% in 2024; 0.92% in 2025). Its timing is a lagged, collection-driven measure of adoption common to all firms (absorbed by year effects); cross-firm differences identify the first stage.

## 2026-09-29 04:56:52 — First outcome results and identification gate G3 (honest record)
- H1 (FE1-C1): beta = 0.0121 (SE 0.00125, t = 9.65), i.e., +1.2 pp experienced-hire share per 1-SD exposure = 3.3% of the pre-period mean. FE2 (NAICS-3 x year): 0.0078 (t = 3.45).
- Event study (FE1-C1): 2016-2018 small positive (0.002-0.0035; t up to 2.4), 2019-2021 flat, 2023 = 0.0125 (t 9.4), 2024 = 0.0081, 2025 = 0.0180. Joint pre-trend test over 2016-2021 p = 0.011 -> G3 threshold (p > 0.10) NOT met in the full window; 2019+ window p = 0.085 (2020 COVID dip t = -1.8).
- Placebo shock dates 2019/2020 in pre-period data: NEGATIVE (t = -2.7, -2.4), i.e., the pre-period drift runs against H1.
- Negative-control (non-language AI) exposure: NEGATIVE post effect (t = -3.6), not null; with both exposures jointly, LLM exposure stays positive (t = 10.3). Interpreted as specificity (the effect is tied to language-model exposure), not as a pass of the null prediction.
- Approximate relative-magnitudes breakdown Mbar* = 1.51.
- Decision (EX POST, labelled as such in the manuscript): add (a) an exposure-specific linear-trend (trend-break) specification, (b) the 2019-2025 window, and (c) the randomization-inference p-value, as supplementary identification evidence. The registered checks are all reported as registered, including the failures.

## 2026-09-29 05:52:08 — Compute plan for world simulations
- Machine shared with other sessions (load average ~200 on 18 cores). Engine made faster without changing results: Frisch-Waugh-Lovell partialling of controls (validated to 1e-15 vs direct fits, intermediate/91_fwl_validation.csv) and a single pass evaluating both policies (validated identical to separate runs on the real data and a null world).
- Registered B kept for Type A (1,000) and Type B (500 per reported moderator). Guard null worlds: 500. Planted worlds REDUCED from the registered 3 beta x 2 gamma x 200 to 3 beta (0.02, 0.05, 0.10 within-SD) x 1 gamma (0.05) x 100 worlds because of compute; reported as a deviation.

## 2026-09-29 06:07:09 — Interim results
- Panel built: 34,333 firm-years, 4,078 firms (2016-2025). Headcount validity r = 0.89 (logs).
- Testbed: H1 beta = 1.21 pp per SD (t = 9.65); IV first-stage F = 562; event study break in 2023; pre-trend p = 0.011 (drift against H1); ex post trend-adjusted estimate 1.38 pp (t = 10.2).
- Real-data loop U: first fully significant package after 3,483 models (entry_rate, FE2, C1, W1; moderators pyramid +, remote +, firm_age -). Ufree: 12 models. H1 alone in that spec t = 2.9; inflated to 10.7 by collinearity with moderator trend terms.
- Guard: exploration half found no full package; its best package's H1 replicated (t = 7.4 on held-out half), moderators mostly not.
- Interim NWC (B = 342): Potemkin U 25.7%, Ufree 91.8%, Bonferroni 0%, spec curve 3.5%; p(H1 max) = 0.003; p(package) = 0.052.
- 2026-09-29 06:40:34: Monte Carlo leakage scenario redefined before its results were used: the first definition (moderator pool chosen by peeking) was uninformative because in the synthetic twin the unguarded loop never assembles a full package under the null (moderators have no link to outcome dynamics in the twin). New definition: after peeking at H1 results across the 12 reduced-grammar specifications, the design space is pruned to the best-looking specification and then "registered"; the pruning is either not replayed (leakage) or replayed (leakage_replayed). Old leakage outputs deleted.

## 2026-09-29 06:56:28 — Type A complete; Type B and ex post diagnostic
- Type A (B = 1,000): Potemkin U 26.5% [23.8, 29.2]; Ufree 93.4%; Bonferroni 0%; spec-curve rule 3.3%; p(H1 max) = 0.001; p(package U) = 0.056.
- Type B (moderator nulls; exact caching validated in intermediate/92_cache_validation.csv): pyramid p = 0.002 (B = 500; permuted pyramid selected in 0.4% of worlds). remote p ~ 0.055 at B = 126, and permuted remote is selected in ~93% of worlds: the within (sector x exposure-quintile) permutation preserves remote's stratum-level link to exposure, so the loop finds 'remote' moderation in null worlds too.
- EX POST diagnostic (tables/expost_nonlinearity.json): adding exposure^2 x Post to the package model shrinks the remote interaction from t = -4.40 to -2.95 (quadratic term t = -0.98); pyramid (-5.21) and firm age (2.30) unchanged.
- MC aggregation bug fixed (leakage glob matched leakage_replayed files); more replications queued (null 200, leakage 300, leakage_replayed 200).

## 2026-09-29 07:43:59 — Final results
- Type A (B=1,000): Potemkin U 26.5%, Ufree 93.4%, Bonferroni package 0%, spec-curve rule 3.3%; Bonferroni on H1 fires in 100% of null worlds; p(H1 max)=0.001; p(package)=0.056.
- Type B (B=500 each): pyramid p=0.002 (Holm 0.006); remote p=0.046 (Holm 0.092); firm age p=0.313.
- Guard null worlds (B=500): package 0.6%, H1 'confirmed' in 48%.
- Planted (3 x 100): NWC package power 98%; Bonferroni 92-93%; loop recovers planted moderators 98.7%; reported/planted main effect median 6.6, 3.2, 2.1 at beta* 0.02, 0.05, 0.10.
- MC (twins): NWC H1 size 5.0% (R=200); pruning-after-peeking not replayed 10.3% (R=300); replayed 5.0% (R=200); confounded 90%; power: package 82%, H1 100%.
- Public synthetic twin regenerated with certified effects (H1, hierarchy shape); pipeline runs with NULLWORLD_DATA=synthetic.

## 2026-10-21 — Validity diagnostics prompted by internal review of the Stage 1 proposal (all EX POST)
- Rule adopted for this round: every analysis below was added after all main results were seen. They are diagnostics of the calibration's conditions and of what its p-values measure; none is used to upgrade a claim. The primary moderator statistic stays the t in the reported package; the primary familywise correction is Holm over all ten registered candidates (already computed by scripts/14 as p_holm_pool10, never reported before).
- Deviation recognized: the registry's Type B entry reads "permute: all moderator candidates independently" and the 2026-09-28 log planned a step-down max-T; scripts/11 ran one-at-a-time worlds for the three reported moderators and the proposal applied Holm over those three. Repaired by joint worlds (scripts/22 joint, B = 1,000). Planned step-down max-T: H2 p = 0.171 (does not confirm H2; reported). Step-down min-P: H2 p = 0.010. Empirical FWER under the global moderator null: Holm over three 14.4%, Holm over ten 4.7%.
- New world constructions (scripts/22): covariate-preserving Type A (exposure keeps its within-sector projection on 12 traits; B = 1,000): H1 p <= 0.001, package p = 0.137, Potemkin U 40.3%. Conditional and Freedman-Lane (null-model) Type B (B = 500 each). Real data: H2 survives all three constructions (Holm-10 0.020); H3 and H4 survive none.
- Monte Carlo (scripts/23, 24): exchangeability dose-response (14 cells x 200 twins x 99 worlds) and moderator constructions (5 scenarios x 200 x 59). Plain permutation over-rejects a zero interaction when the moderator proxies a correlated moderator or nonlinear exposure (35-95%); conditional permutation holds size (4.5-6%); null-model worlds are conservative (0.5%).
- Figure 1 of the proposal: the Output panel had paired the package's H1 t (10.6) with the p of the largest H1 t (12.1); corrected (scripts/21).
- Checks: scripts/28 (57/57 numbers recomputed independently); scripts/27 benchmark (2.5 s per full replay vs ~182 s with pyfixest).

## 2026-10-27/28 — Theory citations for H2 and H3; reproduction notes; public repository
- The H2 and H3 mechanisms, which the registry (frozen 2026-09-29) stated without a citation, are attributed to the theories they apply: knowledge hierarchies (Garicano 2000) for hierarchy shape and task-based technological change (Autor et al. 2003) for digitizable work. No hypothesis, direction, measure, or test changed.
- Descriptive check (not a test, not reported in the proposal): the exposure effect by moderator tercile has the high-minus-low contrast in the registered direction for all three reported moderators (nominal t 1.7-2.2) and is non-monotone for hierarchy shape and digitizable work; re-estimating the reported package from the panel reproduces the stored t statistics (max |dt| 3.3e-14).
- Reproduction section of the technical appendix corrected against the code: the loop and all null worlds (scripts 10-11) run on the synthetic twin; scripts 01-09, 12, 14, 22, 25, and 27 read the licensed data.
- Public repository: https://github.com/LLMPlus-ai/potemkin-findings. Licensed extracts are not included.
- Regression tables (scripts/16, technical appendix Tables A4-A6): fitted to the text width; the column headers of Table A5 now name the hypothesis each column adds (+H4, +H3, +H2, in the order the loop reported the moderators; they had read +H2, +H3, +H4); the "% of pre-period mean" cell is left blank when the pre-period mean is zero (first stage of Table A6). Estimates unchanged: regenerating the tables reproduces every coefficient, standard error, and p-value. The appendix prints the registry without its authoring note (lines 11-14); the hash refers to the complete file.
- Table A11 of the technical appendix (scripts/19): its Holm column now gives Holm over all ten registered moderators (0.020, 0.413, 1.000 for H2, H3, H4; from tables/validity.json), as Table A17 and the proposal do; it had shown Holm over the three reported moderators (0.006, 0.092, 0.313). The search-adjusted p-values were already the same in both tables.
