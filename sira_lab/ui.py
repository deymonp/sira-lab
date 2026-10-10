"""Gradio UI. Works in Colab (share=True) and, later, on a ZeroGPU Space."""
from __future__ import annotations

from typing import Callable, Optional

from .diffview import word_diff

NOTICE = (
    "Research tool for studying text-watermark robustness. Rewriting can change meaning: "
    "always review the output. Do not use it to misrepresent authorship where disclosure is required."
)


def build_ui(rewriter, gpu: Optional[Callable] = None, title: str = "SIRA Lab"):
    import gradio as gr

    gpu = gpu or (lambda f: f)
    cfg = rewriter.cfg

    @gpu
    def run_rewrite(text, ratio, rounds, temp, progress=gr.Progress()):
        if not text or not text.strip():
            return "", "Paste some text first.", []
        try:
            rep = rewriter.rewrite(
                text, mask_ratio=ratio, rounds=int(rounds), temperature=temp,
                progress=lambda f, d: progress(f, desc=d))
        except ValueError as e:
            return "", f"**Error:** {e}", []
        return rep.text_out, rep.to_markdown(), word_diff(rep.text_in, rep.text_out)

    @gpu
    def run_preview(text, ratio):
        if not text or not text.strip():
            return []
        return rewriter.heatmap(text, ratio)

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{NOTICE}")
        with gr.Tabs():
            with gr.Tab("Rewrite"):
                inp = gr.Textbox(label=f"Input (up to {cfg.max_input_words} words)", lines=12)
                with gr.Row():
                    ratio = gr.Slider(0.05, 0.6, value=cfg.mask.mask_ratio, step=0.05, label="Mask ratio")
                    rounds = gr.Slider(1, 3, value=cfg.rounds, step=1, label="Rounds")
                    temp = gr.Slider(0.0, 1.2, value=cfg.gen.temperature, step=0.1, label="Temperature")
                btn = gr.Button("Rewrite", variant="primary")
                out = gr.Textbox(label="Rewritten text", lines=12)
                metrics = gr.Markdown()
                diff = gr.HighlightedText(
                    label="Diff (red = removed, green = added)", combine_adjacent=True, show_legend=True,
                    color_map={"removed": "#f87171", "added": "#4ade80"})
                btn.click(run_rewrite, [inp, ratio, rounds, temp], [out, metrics, diff])
            with gr.Tab("Surprisal preview"):
                inp2 = gr.Textbox(label="Input", lines=10)
                ratio2 = gr.Slider(0.05, 0.6, value=cfg.mask.mask_ratio, step=0.05, label="Mask ratio")
                btn2 = gr.Button("Show high-surprisal words")
                heat = gr.HighlightedText(
                    label="red = would be masked, amber = high self-information", combine_adjacent=True,
                    show_legend=True, color_map={"masked": "#f87171", "high": "#fbbf24"})
                btn2.click(run_preview, [inp2, ratio2], heat)
    return demo
