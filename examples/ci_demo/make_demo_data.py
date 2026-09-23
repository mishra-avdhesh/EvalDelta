"""Create the small, free CI demo: a golden set, cached old results and a 'buggy' candidate.

    python examples/ci_demo/make_demo_data.py

The candidate is a plain Python function (examples/ci_demo/candidate.py) with a planted bug in
one slice, so the example exercises the *live* provider path without any external API.
"""

from pathlib import Path

import pandas as pd

from evaldelta.bench.synthetic import generate_episode

HERE = Path(__file__).parent


def main() -> None:
    ep = generate_episode("slice_regression", n_items=3000, seed=2026, slice_severity=0.25)
    items = ep.items.copy()
    new_loss = items["sample_id"].map(ep._hidden)
    # labels are 'A'/'B'; the old system's answer is correct iff old_loss == 0
    items["reference_answer"] = ["A" if i % 2 else "B" for i in range(len(items))]
    items["old_output"] = [
        ref if ol == 0 else ("B" if ref == "A" else "A")
        for ref, ol in zip(items["reference_answer"], items["old_loss"], strict=True)
    ]
    items.to_parquet(HERE / "items.parquet", index=False)
    # private answer key the toy candidate consults (stands in for a real model's behaviour)
    wrong = new_loss == 1.0
    cand = [
        ("B" if ref == "A" else "A") if w else ref
        for ref, w in zip(items["reference_answer"], wrong, strict=True)
    ]
    pd.DataFrame({"sample_id": items["sample_id"], "answer": cand}).to_parquet(
        HERE / "candidate_behaviour.parquet", index=False
    )
    pd.DataFrame({"sample_id": items["sample_id"], "new_loss": new_loss}).to_parquet(
        HERE / "replay_outcomes_private.parquet", index=False
    )
    print(
        "demo data written; target slice:", ep.meta["target_slice"], "truth:", ep.truth()["delta"]
    )


if __name__ == "__main__":
    main()
