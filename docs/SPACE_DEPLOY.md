# Deploy the CPU-only offline replay demo

The repository's Gradio app is in `apps/hf_space/app.py`. Hugging Face Spaces reads the **root** `README.md` metadata and root `requirements.txt`; the bundled Space creates that layout without including research datasets or trial records. See the [official Spaces configuration](https://huggingface.co/docs/hub/spaces-config-reference) and [Gradio Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-gradio).

```bash
python scripts/build_space_bundle.py
cd dist/space
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`, choose a bundled scenario and budget, and verify the comparison renders. This is an **offline replay**, including when a CSV/Parquet file is uploaded. The app does not execute uploaded code. Do not upload personal data; retention is controlled by the Space host.

After creating a **Gradio** Space under your Hugging Face account, authenticate using the account instructions and copy the bundle into the new Space repository (replace `USERNAME` with yours):

```bash
# Run from the EvalDelta project root after building the bundle.
git clone https://huggingface.co/spaces/USERNAME/evaldelta /tmp/evaldelta-space
cp -a dist/space/. /tmp/evaldelta-space/
cd /tmp/evaldelta-space
git add README.md app.py pyproject.toml requirements.txt LICENSE src .gitignore
git commit -m "Deploy EvalDelta offline replay demo"
git push
```

The GitHub repository and Space are separate Git repositories. Verify the Space build log, run a bundled synthetic scenario on its public URL, and add that URL to the GitHub README. No third-party benchmark matrices or checkpoints are in the bundle. The Space is optional and does not affect research results.
