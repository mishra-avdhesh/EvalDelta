# EvalDelta — Statistical Protocol (v1, frozen before experiments)

This document is the single source of truth for what EvalDelta estimates, how it decides, and what each
decision guarantees. Code in `src/evaldelta/statistics/` implements exactly this; tests in
`tests/statistical/` check it. Changes require a version bump of this file **and** of
`evaldelta.PROTOCOL_VERSION`.

## 1. Paired loss and sign convention

For item `i` in a fixed pool `P` of `N` items, `A` is the baseline (old, production) system and `B` is the
candidate (new) system. The metric adapter produces bounded losses `l_A(i), l_B(i) ∈ [0, 1]`. Lower is better.

    d_i = l_B(i) − l_A(i) ∈ [−1, 1]           (positive = candidate worse = degradation)
    Δ    = (1/N) Σ_{i∈P} d_i                   (pool paired mean difference)

For 0/1 losses (`l = 1 − correct`) each item is one of four cells:

| old \ new | new correct | new wrong |
|-----------|-------------|-----------|
| **old correct** | concordant (d=0) | **down** / negative flip (d=+1) |
| **old wrong**   | **up** / positive flip (d=−1) | concordant (d=0) |

Both discordance directions are always recorded (`n_down`, `n_up`). Swapping `A` and `B` negates every
`d_i` and therefore `Δ`. Continuous metrics must be mapped to `[0,1]` by an explicit, versioned metric
definition. Out-of-range values raise errors; they are never silently clipped.

## 2. Hypotheses and decisions (predeclared per run)

* **Regression (global):** `H0: Δ ≤ δ` vs `H1: Δ > δ`, where `δ ≥ 0` is `regression_margin`.
  The default is `δ = 0`, meaning any detectable worsening. We reject `H0` iff the one-sided lower
  confidence bound `L > δ`. The result is `confirmed_regression`.
* **Non-inferiority (global):** `H0': Δ ≥ δ_NI` vs `H1': Δ < δ_NI`, with `δ_NI > 0` (`noninferiority_margin`).
  We reject `H0'` iff the one-sided upper bound `U < δ_NI`. The result is `evidence_of_noninferiority`.
  This is a separate test, not "failed to reject regression".
* If neither boundary is crossed within the budget, the result is **`inconclusive`**.
  `inconclusive` is never a pass.
* `evaluation_error`: configuration or data errors, or too many failed candidate calls.

**Precedence:** if both boundaries are crossed at the same look, `confirmed_regression` is reported and the
non-inferiority bound is still shown.

**Error guarantees** (under the assumptions in §4):
P(`confirmed_regression` | Δ ≤ δ) ≤ α and P(`evidence_of_noninferiority` | Δ ≥ δ_NI) ≤ α.
These are two different error types, and each is controlled at α.

**Slices.** For each tested slice `g` (predeclared, or selected by discovery; at most `max_slices = G`), the
slice estimand is `Δ_g` and the hypothesis is `H0_g: Δ_g ≤ δ_g`. The slice tests form **one family**.
Its familywise error rate is controlled at `α_slice` (default 0.05) by Holm's procedure across the `G` slices,
combined with the look-level alpha allocation of §5. The global family and the slice family are controlled
**separately**. EvalDelta does not claim a joint FWER across both families unless `joint_fwer=True`, which
splits α equally between them.

## 3. Sealed partition (discovery vs confirmation)

Before any candidate outcome is revealed, the item IDs are sorted and permuted with
`numpy.random.default_rng(split_seed)`. Items are then assigned by position:

* `D` (discovery, default 60%)
* `C_global` (global confirmation, default 20%)
* `C_slice` (slice confirmation, default 20%)

The permutation order inside `C_global` and `C_slice` is the **pre-randomised confirmation order**. The
assignment depends only on the sorted ID set and the seed, so re-ordering input rows cannot change it.
`split_hash = sha256(seed, fractions, ordered IDs per partition)` is stored in every report.

The discovery policy sees public item metadata, cached old-version results, historical results for *other*
version pairs, and candidate outcomes it has **already paid for in `D`**. It never sees candidate outcomes in
`C_global`, in `C_slice`, or for unqueried items in `D`. This is enforced by the replay oracle
(`evaldelta.replay.oracle`). Policies receive a `PolicyView` that has no reference to the oracle.

## 4. Confirmation estimands and validity arguments

**Global (OURS-1 and all fixed-confirmation baselines).** The global confirmation plan (sample size
`n_C`, looks, α per look, method) is fixed in the config **before** the run. It never depends on discovery data.
Confirmation items are taken from `C_global` in its pre-randomised order. `C_global` is a uniformly random
subset of the pool, and its internal order is a uniform permutation. The first `n` confirmation items are
therefore a simple random sample **without replacement (SRSWoR)** from the whole pool `P`. The global test uses
only this sample, so its validity does not depend on anything discovery did.

* `betting_wor` (default for non-zero margins and non-binary losses): the finite-population betting
  confidence sequence (Waudby-Smith & Ramdas 2023, §6) for `Δ` over `P` (population size `N`). It is
  **exact for the pool estimand** and anytime-valid (Ville's inequality).
* `mcnemar_exact` (default when losses are 0/1 and `δ = 0`): one-sided exact binomial test on the discordant
  pairs, `p = P(Bin(n_down+n_up, ½) ≥ n_down)`. Its sampling assumption is that confirmation pairs are
  i.i.d. draws from the target distribution (superpopulation estimand `E[d]`). This holds when the golden set
  is an i.i.d. sample from the target distribution, because a randomly chosen subset of an i.i.d. sample is
  i.i.d. For the finite-pool estimand, `tests/statistical/test_mcnemar_finite_pool.py` checks the size of
  the test under exact hypergeometric sampling at the boundary `n_down_pool = n_up_pool`. It is reported
  as a numerical check, not a theorem. The compatible interval is the one-sided Clopper–Pearson bound for
  `π = P(down | discordant)`. McNemar rejects iff that bound is `> ½`. The effect size `Δ̂ = (n_down − n_up)/n`
  is reported together with the `betting_wor` interval for `Δ`, which is labelled separately.
* `hoeffding_wor`: Hoeffding's inequality (valid for SRSWoR, Hoeffding 1963, Thm 4), range 2. It is a
  conservative reference only.

**Slices.** A slice `g` may be selected using discovery data. Its confirmation sample is taken from
`C_slice ∩ g` in pre-randomised order. The order inside `C_slice` is independent of `D` and of the selection.
So, conditional on the selection, the sample is an SRSWoR from `C_slice ∩ g`. The **exact** slice estimand is
therefore `Δ_g^{C}`, the mean over the slice's members in the slice-confirmation partition (a uniformly
random ~20% subsample of the slice). The reported interpretation as the slice mean `Δ_g` rests on that
random-subsample relationship, or on the i.i.d. superpopulation model. A slice with fewer than
`min_slice_confirm` (default 30) fresh items in `C_slice ∩ g` is reported as
`inconclusive: insufficient_fresh_examples` and is never confirmed.

## 5. Looks and alpha allocation

A plan declares `K` looks at predeclared confirmation counts `n_1 < … < n_K = n_C`. The default is `K = 3`,
at `⌈n_C·k/K⌉`.

* Fixed-sample methods (`mcnemar_exact`, `hoeffding_wor`) use **Bonferroni** with `α/K` per look.
* Betting confidence sequences are anytime-valid. The running maximum of the capital process is compared
  with `1/α`, so all looks share α without splitting. This is the "advanced, separately tested" path in the
  specification. Its calibration is shown in `docs/CALIBRATION.md`.
* The run stops at the first look where a decision boundary is crossed. Any remaining budget is reported
  as `unspent`.
* An uncorrected repeated fixed-sample p-value is never reported as anytime-valid.
  `tests/statistical/test_optional_stopping.py` shows that naive peeking inflates the error, and that the
  corrected plans do not.

Slices use Holm across the `G` tested slices, applied to per-slice p-values or e-values. Each slice's p-value
is its minimum over looks, multiplied by `K` (Bonferroni over looks). For betting slices the p-value is
`1 / max_t K_t` (Ville).

## 6. Betting confidence sequence (implementation contract)

Observations `Y_s` with a predictable lower bound `L_s`, and a conditional mean under the hypothesised value `m`
that is affine and increasing in `m`: `μ_s(m) = a_s + b_s·m`, with `b_s > 0`.

    K_t(m) = Π_{s≤t} (1 + λ_s(m)·(Y_s − μ_s(m))),   λ_s(m) = min(λ̃_s, c/(μ_s(m) − L_s)),  c = 0.5

`λ̃_s ≥ 0` is predictable. It is the predictable plug-in `sqrt(2·log(1/α)/(n_plan·σ̂²_{s−1}))`, with a
regularised running variance, independent of `m`. Every factor is `≥ 1 − c > 0`. Each factor is nonincreasing
in `m` (shown in the code docstring), so the rejection region `{m : max_{s≤t} K_s(m) ≥ 1/α}` is a lower
half-line. The lower bound is found by bisection. At the true `m`, `K_t` is a nonnegative martingale, so Ville
gives `P(∃t: K_t(Δ) ≥ 1/α) ≤ α`. Values of `m` that are logically impossible given the observed data
(`μ_s(m) < L_s`, i.e. the remaining mean would fall below its bound) are rejected. The true value is never
impossible. The upper bound is the lower bound of `−Y`.

* i.i.d./with replacement: `a=0, b=1`, `L=−1`.
* WoR from a population of size `N` (in `d` units): `μ_s(m) = (N·m − S_{s−1}) / (N − s + 1)`, `L = −1`.

## 7. Research extension (OURS-2): propensity-weighted adaptive global inference

This is a separate, opt-in method: `AdaptiveGlobalPlan`. It is **not** the default. At step `t` the policy
(using only already-revealed outcomes) outputs a predictor `ĝ_t(i)` of `d_i` and sampling probabilities
`q_t(i)` over the not-yet-revealed eligible items `R_t`, with `q_t(i) ≥ ε/|R_t|`. For already-revealed items
`d_i` is known exactly, so `ĝ_t(i) = d_i` and `q_t(i) = 0`. Draw `I_t ~ q_t` and pay for it. Define

    Y_t = (1/N) Σ_{i∈P} ĝ_t(i) + (d_{I_t} − ĝ_t(I_t)) / (N·q_t(I_t))

**Unbiasedness:** `E[Y_t | F_{t−1}] = (1/N)Σ_i ĝ_t(i) + (1/N)Σ_{i∈R_t}(d_i − ĝ_t(i)) = Δ`. This holds
because revealed items contribute zero residual.

**Bounds:** with `d_i ∈ [lo_i, hi_i]` (for 0/1 losses and a cached old loss, `[−l_A, 1−l_A]`), the lower bound
`L_t = ḡ_t + min_{i∈R_t}(lo_i − ĝ_t(i))/(N q_t(i))` is predictable. Applying §6 with `a=0, b=1` gives an
anytime-valid CS for the pool `Δ` under **arbitrary** adaptive (predictable) `q_t` and `ĝ_t`. A bad predictor
harms efficiency but not validity. Every paid call reveals a new item, so draws are equal to paid calls.
The specification's conservative Hoeffding radius (`Z_t ∈ [−1/ε, 1/ε]`, union over t) is implemented as a
reference (`hoeffding_ipw`).

Propensities `q_t(I_t)` are logged for every draw (mandatory in propensity mode). Deterministic rank-based
discovery never reports invented propensities.

## 8. Budget accounting

* The **cost unit** is declared per run: `candidate_calls` (default), or a numeric `estimated_candidate_cost`
  column in declared units (e.g., tokens, USD, GPU-seconds). Units are never mixed.
* Only candidate evaluations are paid. Old-version results are cached and free unless `old_cached=false`, in
  which case old evaluations are charged in the same unit.
* Partitioning and feature computation are free in the budget. Their wall time is still measured and
  reported.
* Every attempt, including retries and failed calls, is charged by default (`charge_failed_attempts=true`).
  Failures are tracked separately from correctness failures, and a failed item is never scored as wrong.
* Phase budgets (`discovery`, `global_confirmation`, `slice_confirmation`; defaults 0.3/0.5/0.2 of the total)
  are fixed ceilings. Unspent budget is reported and is not silently moved to another phase.
* `BudgetExceeded` is raised **before** any call that would exceed a ceiling. The total can never exceed the
  declared maximum; property-based tests check this.

## 9. What the protocol does not guarantee

* Absence of a confirmed regression is not evidence of safety. Only `evidence_of_noninferiority` against a
  declared margin supports approval, and only for the declared estimand and population.
* Rare slices below `min_slice_confirm` cannot be confirmed at any budget within the partition.
* Validity assumes the old results are genuinely cached from the old system on the same items, that
  the metric adapter is deterministic, and that the candidate is evaluated once per item. For stochastic
  candidates, `d_i` becomes random. The betting and McNemar i.i.d. arguments still hold for the
  superpopulation, but not the finite-pool exactness.
* Reusing the same golden set across many releases causes adaptive overfitting. v1 warns when a set has been
  used in more than 20 recorded runs.
