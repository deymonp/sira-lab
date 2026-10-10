"""Launcher. Colab: `!python app/app.py --share`.  CPU/UI smoke test: `python app/app.py --fake`."""
import argparse

from sira_lab.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fake", action="store_true", help="use fake models (no downloads, UI test only)")
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fake:
        from sira_lab.testing import build_fake_rewriter

        rewriter = build_fake_rewriter(cfg)
    else:
        from sira_lab.factory import build_rewriter

        rewriter = build_rewriter(cfg)
    from sira_lab.ui import build_ui

    build_ui(rewriter).queue().launch(share=args.share)


if __name__ == "__main__":
    main()
