# SIRA Lab

Research toolkit for studying **text-watermark robustness**: a SIRA-style rewriter
(Self-Information Rewrite Attack, Cheng et al., ICML 2025, arXiv 2505.05190) plus an
evaluation lab (in progress). Zero-cost stack: Google Colab (free), Hugging Face Hub/Spaces, Gradio.

> **Intended use:** robustness research and false-positive analysis of AI-text detectors.
> Rewriting can change meaning, so always review outputs. Do not use it to misrepresent
> authorship where disclosure is required.

## Status

| Part | State |
| --- | --- |
| Rewriter core (normalize, chunk, score, mask, infill, gates) | Implemented; unit-tested with fake models; real-model path first exercised in Colab |
| Gradio UI (rewrite + surprisal preview) | Implemented; smoke-tested with fake models |
| Evaluation Lab (MarkLLM watermark generation, detector panel) | Next (metrics module done) |
| ZeroGPU Space deployment | Planned (Sprint 4) |

## How it works

1. **Normalize** Unicode (zero-width characters, ligatures, look-alike letters).
2. **Chunk** into paragraphs of at most 250 words; headings, captions, references and equations are passed through untouched.
3. **Score** every token's self-information with a small open LM (`-log p(token | context)`, in nats).
4. **Mask** the words with the *highest* self-information (the high-entropy positions where watermark signal tends to sit),
   never touching citations, numbers, acronyms, names, quotes, math or code.
5. **Infill** the masks with a different open-weights instruct LLM. Default `gen.mode: list`: the model returns only
   `N: word` lines and we insert them ourselves, so text outside the masks is never altered and generation is fast.
   If a model ignores that format, the chunk falls back to `passage` mode (model rewrites the whole passage; strict alignment check).
   Every failed attempt records what the model actually said in `ChunkReport.warnings`.
6. **Gate**: semantic similarity to the original (default >= 0.85) and perplexity ratio (<= 1.6). Failed chunks retry with a lower mask ratio.

## Quick start (Colab, free T4)

1. Open `notebooks/00_setup.ipynb` in Colab (File -> Open notebook -> GitHub tab -> `deymonp/sira-lab`), set the runtime to **T4 GPU**.
2. Add Colab secret `HF_TOKEN` (key icon, **Notebook access** on).
3. Run all cells; expect `34 passed`.
4. Open `notebooks/01_quickstart.ipynb` and run it.

CPU-only UI check (no downloads): `python app/app.py --fake`

## Layout

```
sira_lab/   normalize.py chunking.py scorer.py masker.py infill.py backends.py gates.py pipeline.py factory.py ui.py testing.py
sira_lab/lab/  metrics.py (generation + detector adapters coming)
configs/    default.yaml (T4)  smoke.yaml (tiny models)  space.yaml (ZeroGPU, untested)
notebooks/  00_setup  01_quickstart
tests/      CPU-only unit tests
app/app.py  launcher
```

## Models and licenses

Defaults: `Qwen/Qwen2.5-0.5B` (scorer), `Qwen/Qwen2.5-7B-Instruct` (infiller), `all-MiniLM-L6-v2` (similarity).
Check each model card's license before redistributing anything derived from them.

## Reproducibility

Every run records a config hash (`Report.config_hash`) and seed. `00_setup` writes `requirements.lock.txt` after a successful run.
