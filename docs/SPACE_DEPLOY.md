# Deploy a demo

## Free static Space

The static demo selects among saved results from deterministic synthetic runs of the EvalDelta Python engine. It does not execute Python in the browser, accept uploads, or claim that a saved run is a live model evaluation. [Static Spaces are free](https://huggingface.co/docs/hub/spaces-overview); no GPU or paid plan is needed.

From the project root, build and inspect the bundle:

```bash
pip install -e .
python scripts/build_static_space.py
python -m http.server 8765 --directory dist/static-space
```

Open `http://localhost:8765`. The generated `data.json` has six scenarios, three budgets, and two selector results per run. The bundle contains only `index.html`, `data.json`, `README.md`, and `LICENSE`.

On Hugging Face, create a **public Static** Space named `evaldelta-demo` under your account. Then copy the bundle into its Git repository, replacing `USERNAME` below. Authenticate with your own write credential if Git asks; do not put a token in the remote URL or commit it.

```bash
git clone https://huggingface.co/spaces/USERNAME/evaldelta-demo /tmp/evaldelta-static-space
cp -a dist/static-space/. /tmp/evaldelta-static-space/
cd /tmp/evaldelta-static-space
git add README.md index.html data.json LICENSE
git commit -m "Publish EvalDelta synthetic replay demo"
git push
```

After the Space builds, open its public App tab, change the scenario and budget, and check both result cards. Add the live Space URL to the main GitHub README in a follow-up commit. The real-data benchmark findings remain in [RESULTS.md](RESULTS.md), separate from these synthetic examples.

## Optional live Gradio replay

The Python-backed app in `apps/hf_space/app.py` can evaluate an uploaded **offline outcome table** or generate a synthetic episode on request. Build its separate data-free bundle with `python scripts/build_space_bundle.py`; run it locally with `python dist/space/app.py` after installing `.[demo]`.

As of September 2026, creating a Gradio Space on CPU Basic requires a paid personal plan. Eligible free personal accounts can host up to two Gradio Spaces on ZeroGPU, subject to Hugging Face's account conditions and runtime rules. Check the [current Spaces overview](https://huggingface.co/docs/hub/spaces-overview) and [ZeroGPU guide](https://huggingface.co/docs/hub/spaces-zerogpu) before choosing this route. The static demo above is the no-cost default.
