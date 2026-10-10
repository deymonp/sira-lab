"""Orchestrates: normalize -> chunk -> score -> mask -> infill -> gates -> reassemble."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

from .chunking import Chunk, count_words, join_chunks, make_chunks
from .config import Config, config_hash
from .gates import check
from .infill import build_messages, parse_candidate
from .masker import MaskResult, Word, find_words, heat_segments, make_template, score_words, select_masks
from .normalize import normalize


@dataclass
class ChunkReport:
    index: int
    status: str                      # skipped:<why> | rewritten | best-effort | unchanged
    orig: str
    out: str
    n_words: int = 0
    n_masked: int = 0
    n_changed: int = 0
    attempts: int = 0
    sim: Optional[float] = None
    ppl_ratio: Optional[float] = None
    mask_ratio_used: Optional[float] = None
    swaps: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Report:
    text_in: str
    text_out: str
    chunks: List[ChunkReport]
    sim: Optional[float]
    ppl_ratio: Optional[float]
    words_total: int
    words_masked: int
    words_changed: int
    elapsed_s: float
    config_hash: str
    normalize_stats: dict
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        sim = f"{self.sim:.3f}" if self.sim is not None else "n/a"
        ppl = f"{self.ppl_ratio:.2f}" if self.ppl_ratio is not None else "n/a"
        lines = [
            f"**Semantic similarity (word-weighted):** {sim}  \n"
            f"**Perplexity ratio (out/in):** {ppl}  \n"
            f"**Words:** {self.words_total} total, {self.words_masked} masked, {self.words_changed} changed  \n"
            f"**Time:** {self.elapsed_s:.1f}s  |  **Config:** `{self.config_hash}`"
        ]
        statuses = {}
        for c in self.chunks:
            statuses[c.status] = statuses.get(c.status, 0) + 1
        lines.append("**Chunks:** " + ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())))
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n\n".join(lines)


@dataclass
class _State:
    chunk: Chunk
    cur: str
    status: str = "skipped"
    n_masked: int = 0
    n_changed: int = 0
    attempts: int = 0
    sim: Optional[float] = None
    ppl_ratio: Optional[float] = None
    ratio_used: Optional[float] = None
    swaps: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # per-round scratch
    ratio: float = 0.0
    words: List[Word] = field(default_factory=list)
    scores: object = None
    best: Optional[tuple] = None
    done: bool = False
    orig_ms: Optional[float] = None


class SiraRewriter:
    def __init__(self, cfg: Config, scorer, backend, sim_model):
        self.cfg, self.scorer, self.backend, self.sim_model = cfg, scorer, backend, sim_model
        try:
            from transformers import set_seed

            set_seed(cfg.seed)
        except Exception:
            pass

    # ------------------------------------------------------------------ preview
    def heatmap(self, text: str, mask_ratio: Optional[float] = None):
        norm, _ = normalize(text)
        segs: list = []
        for ch in make_chunks(norm, self.cfg.chunk):
            if segs:
                segs.append(("\n\n", None))
            if not ch.rewrite:
                segs.append((ch.text, None))
                continue
            ts = self.scorer.score(ch.text)
            words = find_words(ch.text, self.cfg.mask)
            score_words(words, ts.spans, ts.surprisal, self.cfg.mask.agg)
            chosen = select_masks(words, self.cfg.mask, mask_ratio)
            segs += heat_segments(ch.text, words, chosen)
        return segs

    # ------------------------------------------------------------------ main
    def rewrite(self, text: str, *, mask_ratio: Optional[float] = None, rounds: Optional[int] = None,
                temperature: Optional[float] = None,
                progress: Optional[Callable[[float, str], None]] = None) -> Report:
        cfg, t0 = self.cfg, time.time()
        if count_words(text) > cfg.max_input_words:
            raise ValueError(f"Input has {count_words(text)} words; limit is {cfg.max_input_words}.")
        ratio0 = cfg.mask.mask_ratio if mask_ratio is None else mask_ratio
        rounds = cfg.rounds if rounds is None else rounds
        temp = cfg.gen.temperature if temperature is None else temperature

        norm, nstats = normalize(text)
        chunks = make_chunks(norm, cfg.chunk)
        states = [_State(c, c.text, status=f"skipped:{c.skip_reason}" if not c.rewrite else "unchanged") for c in chunks]
        active = [s for s in states if s.chunk.rewrite]
        for r in range(rounds):
            self._run_round(active, r, ratio0, temp, progress, rounds)

        out = join_chunks(chunks, [s.cur for s in states])
        reports, wsum, simsum, pplsum = [], 0, 0.0, 0.0
        for s in states:
            n_words = count_words(s.chunk.text)
            reports.append(ChunkReport(
                s.chunk.index, s.status, s.chunk.text, s.cur, n_words, s.n_masked, s.n_changed,
                s.attempts, s.sim, s.ppl_ratio, s.ratio_used, s.swaps, s.warnings))
            if s.chunk.rewrite and s.sim is not None:
                wsum += n_words
                simsum += n_words * s.sim
                pplsum += n_words * (s.ppl_ratio or 1.0)
        warnings = []
        n_best = sum(r.status == "best-effort" for r in reports)
        if n_best:
            warnings.append(f"{n_best} chunk(s) failed the quality gates; the closest candidate was kept. Review them.")
        if any(r.status == "unchanged" for r in reports):
            warnings.append("Some chunks could not be rewritten and were left unchanged.")
        return Report(
            text_in=text, text_out=out, chunks=reports,
            sim=simsum / wsum if wsum else None, ppl_ratio=pplsum / wsum if wsum else None,
            words_total=sum(r.n_words for r in reports),
            words_masked=sum(r.n_masked for r in reports),
            words_changed=sum(r.n_changed for r in reports),
            elapsed_s=time.time() - t0, config_hash=config_hash(cfg),
            normalize_stats=nstats, warnings=warnings)

    # ------------------------------------------------------------------ internals
    def _run_round(self, states: List[_State], r: int, ratio0: float, temp: float, progress, rounds: int):
        cfg = self.cfg
        for s in states:
            s.ratio, s.best, s.done = ratio0, None, False
            s.scores = self.scorer.score(s.cur)
            s.words = find_words(s.cur, cfg.mask)
            score_words(s.words, s.scores.spans, s.scores.surprisal, cfg.mask.agg)
            if s.orig_ms is None:
                s.orig_ms = self.scorer.mean_surprisal(s.chunk.text)
        pending = list(states)
        prev_text = {id(s): s.cur for s in states}

        for attempt in range(cfg.gen.max_attempts):
            if not pending:
                break
            prepared: List[Tuple[_State, MaskResult]] = []
            for s in pending:
                chosen = select_masks(s.words, cfg.mask, s.ratio)
                if not chosen:
                    s.warnings.append("no maskable words")
                    s.done = True
                    continue
                prepared.append((s, make_template(s.cur, chosen, s.words)))
            pending = [s for s in pending if not s.done]
            if not prepared:
                break
            batch = [build_messages(mr.template) for _, mr in prepared]
            longest = max(count_words(mr.template) for _, mr in prepared)
            max_new = min(cfg.gen.max_new_tokens, int(1.7 * longest) + 120)
            raws = self.backend.generate(batch, max_new, temp, cfg.gen.top_p)

            nxt: List[_State] = []
            for (s, mr), raw in zip(prepared, raws):
                s.attempts += 1
                cand = parse_candidate(mr.template, [m.original for m in mr.masks], raw, cfg.gen.strict)
                if cand is None:
                    nxt.append(s)
                    continue
                text_c, swaps = cand
                g = check(s.chunk.text, text_c, s.orig_ms, self.sim_model, self.scorer, cfg.gate)
                if s.best is None or g.sim > s.best[0]:
                    s.best = (g.sim, g.ppl_ratio, text_c, swaps, len(mr.masks), s.ratio)
                if g.ok:
                    self._accept(s, g.sim, g.ppl_ratio, text_c, swaps, len(mr.masks), s.ratio, "rewritten")
                else:
                    s.ratio *= cfg.gate.ratio_decay
                    s.warnings.append(f"round {r + 1} attempt {attempt + 1}: " + "; ".join(g.reasons))
                    nxt.append(s)
            pending = nxt
            if progress:
                progress((r + (attempt + 1) / cfg.gen.max_attempts) / rounds, f"round {r + 1}/{rounds}")

        for s in pending:  # exhausted attempts without passing the gates
            if s.done:
                continue
            had_earlier = s.status == "rewritten"
            if had_earlier:
                s.warnings.append(f"round {r + 1} failed; kept the result of the previous round")
            elif s.best is not None and cfg.gate.on_fail == "best":
                sim, ppl, text_c, swaps, nm, ratio = s.best
                self._accept(s, sim, ppl, text_c, swaps, nm, ratio, "best-effort")
            else:
                s.cur = prev_text[id(s)]
                s.status = "unchanged"
                s.warnings.append("could not produce a valid rewrite")

    @staticmethod
    def _accept(s: _State, sim, ppl, text_c, swaps, n_masked, ratio, status):
        s.cur, s.sim, s.ppl_ratio, s.swaps = text_c, sim, ppl, swaps
        s.n_masked, s.ratio_used, s.status, s.done = n_masked, ratio, status, True
        s.n_changed = sum(1 for o, f in swaps if o.lower() != f.lower())
