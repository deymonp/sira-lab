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
| Evaluation Lab: watermarks (KGW, Unigram, EXP, SynthID-Text), attacks (SIRA, random, lowest, paraphrase), detector panel (watermark z-tests, Fast-DetectGPT, Binoculars), TPR@FPR/AUROC, trade-off plot | Implemented; maths and flow unit-tested with numpy toys; GPU stages first run in Colab (`notebooks/02_lab.ipynb`). See `ROADMAP.md` |
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
3. Run all cells; expect `86 passed`.
4. Open `notebooks/01_quickstart.ipynb` and run it.

CPU-only UI check (no downloads): `python app/app.py --fake`

## Layout

```
sira_lab/   normalize.py chunking.py scorer.py masker.py infill.py backends.py gates.py pipeline.py factory.py ui.py testing.py
sira_lab/lab/  analysis.py judge.py select.py watermark.py detectors.py zeroshot_math.py generate.py attacks.py evaluate.py metrics.py prompts.py run.py io.py
configs/    default.yaml (T4)  smoke.yaml (tiny)  space.yaml (ZeroGPU, untested)  lab_smoke / lab_quick / lab_full / lab_dose / lab_v7 / lab_v8 .yaml
notebooks/  00_setup  01_quickstart  02_lab  03_dose  04_v7  05_v8
tests/      CPU-only unit tests
app/app.py  launcher
```

## Models and licenses

Defaults: `Qwen/Qwen2.5-0.5B` (scorer), `Qwen/Qwen2.5-7B-Instruct` (infiller), `all-MiniLM-L6-v2` (similarity).
Check each model card's license before redistributing anything derived from them.

## Reproducibility

Every run records a config hash (`Report.config_hash`) and seed. `00_setup` writes `requirements.lock.txt` after a successful run.

## Evaluation Lab

`notebooks/02_lab.ipynb` runs three resumable stages (outputs are JSONL/CSV files in a run folder):

1. **generate**: human reference (arXiv abstracts), unwatermarked and watermarked continuations from an open LM.
2. **attack**: SIRA vs. control strategies (random words, lowest-surprisal words) vs. a plain LLM paraphrase.
3. **evaluate**: z-score detection, TPR at a fixed FPR, AUROC, semantic similarity, perplexity ratio, plus a trade-off plot.

Watermarks: `kgw`, `unigram`, `exp` are our own implementations (same statistics as the papers, different
pseudo-random function); `synthid` uses the SynthID-Text reference implementation in `transformers` with a mean-g-value
detector (not Google's production Gemini watermark). Detectors: the matching watermark z-tests plus Fast-DetectGPT and
Binoculars (zero-shot, "is it machine-written?"). Watermark detectors are calibrated on unwatermarked machine text;
zero-shot detectors on human text. Not yet covered: SIR / EWD / SWEET / DiPmark / Christ (via MarkLLM, phase 4),
classifier-based detectors. See `ROADMAP.md`.

## Matched edit budget (v6)

`notebooks/03_dose.ipynb` + `configs/lab_dose.yaml` compare SIRA with the random and lowest-surprisal controls at the same
share of changed words. Attack names are `strategy@mask_ratio`, with an optional `xN` for N rewrite rounds
(`sira@0.3`, `random@0.15`, `sira@0.3x2`). With `gate.enforce: false` the similarity and perplexity gates only measure.
Evaluation also writes `scores.csv` (one detector score per text), `paired.csv` (per-text paired comparisons with bootstrap
95% intervals) and `dose_response.png`. Note: only about 46% of words are maskable (stopwords, names, numbers, citations are
protected), so SIRA and random converge above a mask ratio of roughly 0.4; use `xN` rounds for larger edit budgets.

## Combined attacks, held-out detectors, quality judge, success rate (v7)

`notebooks/04_v7.ipynb` + `configs/lab_v7.yaml`.
* **Chains**: `A+B` applies attack A then B (`paraphrase+random@0.15`). A finished `attacked_<A>.jsonl` is re-used as stage 1.
* **Held-out detectors** `fast_detectgpt_ho`, `binoculars_ho`: SmolLM2 (a different model family than the Qwen generator).
* **Quality judge** (`lab/judge.py`): the 7B model rates fluency of every text and meaning preservation of every rewrite (1 to 5),
  and also rates untouched human and machine text as reference points. Caveat: a 7B judge of the same family as the rewriter is
  noisy and may favour its own style; it is a better check than embedding similarity, not a human reader.
* **Success rate** (`analysis.success_table`): the share of texts that beat EVERY detector AND have fluency and meaning >= 4.
* **Score caching**: `evaluate` re-uses `scores.csv` (a changed text is re-scored), skips loading a detector whose scores are all
  cached, and saves after every detector, so an interrupted evaluation resumes.

## Fill hygiene and best-of-n selection (v8)

`notebooks/05_v8.ipynb` + `configs/lab_v8.yaml`.
* **Fill hygiene** (attack families `sira2`, `random2`, `lowest2`; `gen.hygiene`): plain-word fills only (no brackets or punctuation),
  fills that merely repeat the neighbouring word are trimmed (or the original word is kept), "a"/"an" agree with the fill,
  and no two masks touch (`min_gap` >= 1). The v7 attacks (`sira`, `random`, ...) keep their old behaviour for comparison.
* **Variants**: `NAME~k` is NAME with another random seed. `prep_attacks` are computed but not scored or judged.
* **Best of n** (`lab/select.py`): per text, keep the variant that the original detectors miss and judge A rates best. It copies
  that variant's scores and judge rows under the name `bestN`, so no detector model is reloaded.
* **Honest grading**: held-out detectors (`*_ho`) take no part in the choice. Judge A takes part in the choice, so the final
  quality grade uses **judge B** (`judge.run_judge_b`: another model family, `microsoft/Phi-3.5-mini-instruct`, another rubric
  wording; falls back to the main model with the second wording if it cannot be loaded). `success_table(quality_file="quality_b.csv")`.
* `success_table` now leaves out attacks that were never judged or lack some detector, instead of showing a misleading 0%.
