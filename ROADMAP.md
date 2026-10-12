# Roadmap

Goal: measure, and then improve, how well text rewriting hides machine-written text from a panel of detectors
(watermark detectors and zero-shot detectors) while keeping the meaning. "Bypass everything" is the target to
measure against, not an assumption: the Lab reports which detectors are beaten, at what cost in meaning, and which are not.

Working success criterion (adjust after the first results): TPR@1%FPR <= 0.10 on a detector, with median
semantic similarity >= 0.85 and no more than 10% failed rewrites.

| Phase | What | State |
| --- | --- | --- |
| 1 | SIRA rewriter (score, mask, infill, gates), random / lowest-surprisal controls, LLM-paraphrase baseline | Built; first Colab runs done with tiny models |
| 2 | Lab: watermarks KGW, Unigram, EXP, SynthID-Text (open reference implementation); detectors Fast-DetectGPT, Binoculars | Built (v5); GPU stages first run in Colab |
| 3 | First measurements. Smoke run and a 40-text quick run done: baselines sound; gates were weakening SIRA (about 15% of words changed instead of 30%); SIRA about equal to random on KGW, slightly better on SynthID within noise; plain paraphrase stronger but changes 68% of words | Done (see phase 3b) |
| 3b | Matched-edit-budget experiment (`lab_dose.yaml`, notebook 03): SIRA vs random vs lowest at equal share of changed words, gates measure-only, paired statistics | v6, next to run |
| 4 | More watermark families through MarkLLM (SIR, EWD, SWEET, DiPmark, Christ, ...): clone it in Colab, check it works with the installed transformers, then add an adapter | After phase 3 |
| 5 | Attack suite v2, designed from the phase-3 numbers (mask ratio / rounds sweeps, SIRA + paraphrase, strategies that raise rather than lower perplexity for zero-shot detectors) | After phase 3 |
| 6 | Evaluation hardening: held-out detectors, keys and generator models (no tuning on the test panel), bootstrap confidence intervals, other text domains, classifier-based detectors | After phase 5 |
| 7 | Public ZeroGPU Space for the rewriter plus a read-only results tab | After phase 6 |

## Known limits (be careful when quoting results)

- SynthID-Text here is the `transformers` reference implementation. Google says its hashing differs from the one
  used for Gemini, so results do not transfer to Gemini output.
- KGW, Unigram and EXP are our own implementations: same statistics as the papers, different pseudo-random function.
- Zero-shot detectors depend on the sampling settings of the text they see. Pure temperature-1 sampling (the setting
  many watermark papers use) is hard for them even before any attack. We generate with temperature 0.8 and top-p 0.95
  and say so in every table.
- Commercial detectors (GPTZero, Turnitin, ...) are not free to query and are out of scope for the zero-cost rule.
- With fewer than ~100 negatives, TPR@1%FPR is a rough number.
