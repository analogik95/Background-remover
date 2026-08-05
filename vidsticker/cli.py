"""Command line interface for vidsticker."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .matting import MODES, MatteConfig
from .pipeline import FORMATS, StickerOptions, create_sticker

MODELS = ("isnet-general-use", "u2net", "u2netp", "u2net_human_seg",
          "isnet-anime", "birefnet-general", "birefnet-portrait")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidsticker",
        description="Turn a video into a transparent animated sticker (GIF / WebP / APNG).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  vidsticker clip.mp4                        # GIF + WebP, auto matte, 512 px
  vidsticker clip.mp4 -o out --size 320      # smaller sticker
  vidsticker clip.mp4 --mode ai              # ignore the backdrop colour
  vidsticker clip.mp4 --formats gif webp apng --square
  vidsticker --serve                         # browser UI on :8000
""")
    p.add_argument("input", nargs="?", type=Path, help="source video")
    p.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory (default: out)")
    p.add_argument("-n", "--name", help="output filename stem (default: input name)")

    g = p.add_argument_group("sticker")
    g.add_argument("--size", type=int, default=512, help="longest edge in px (default: 512)")
    g.add_argument("--fps", type=float, help="output frame rate (default: source, snapped for GIF)")
    g.add_argument("--formats", nargs="+", default=["gif", "webp"], choices=FORMATS,
                   help="outputs to write (default: gif webp)")
    g.add_argument("--square", action="store_true", help="pad the crop to a square canvas")
    g.add_argument("--no-crop", dest="crop", action="store_false", help="keep the original framing")
    g.add_argument("--pad", type=int, default=6, help="px of margin around the subject (default: 6)")
    g.add_argument("--loop", type=int, default=0, help="loop count, 0 = forever (default: 0)")
    g.add_argument("--start", type=float, help="trim start in seconds")
    g.add_argument("--duration", type=float, help="trim length in seconds")

    m = p.add_argument_group("matting")
    m.add_argument("--mode", choices=MODES, default="auto",
                   help="auto picks hybrid on a coloured screen, ai otherwise (default: auto)")
    m.add_argument("--model", default="isnet-general-use", help=f"rembg model (e.g. {', '.join(MODELS[:4])})")
    m.add_argument("--key", nargs=2, type=float, metavar=("CR", "CB"),
                   help="force the chroma key colour instead of detecting it")
    m.add_argument("--tolerance", type=float, default=8.0, help="chroma radius kept fully transparent (default: 8)")
    m.add_argument("--softness", type=float, default=20.0, help="chroma edge ramp width (default: 20)")
    m.add_argument("--despill", type=float, default=0.8, help="screen-colour spill removal, 0..1 (default: 0.8)")
    m.add_argument("--feather", type=float, default=0.0, help="blur the matte edge by N px")
    m.add_argument("--shrink", type=float, default=0.0, help="erode the matte by N px to bite off fringing")
    m.add_argument("--no-smooth", dest="smooth", action="store_false",
                   help="disable the 3-frame temporal median")
    m.add_argument("--work-size", type=int, default=1280,
                   help="internal matting resolution, longest edge (default: 1280)")

    e = p.add_argument_group("encoding")
    e.add_argument("--alpha-threshold", type=int, default=128,
                   help="GIF alpha cutoff, 1..255 (default: 128)")
    e.add_argument("--dither", default="sierra2_4a",
                   choices=["sierra2_4a", "bayer", "floyd_steinberg", "none"],
                   help="GIF dithering (default: sierra2_4a)")
    e.add_argument("--colors", type=int, default=255, help="GIF palette size, max 255 (default: 255)")
    e.add_argument("--webp-quality", type=int, default=90, help="WebP quality 0-100 (default: 90)")
    e.add_argument("--no-preview", dest="preview", action="store_false",
                   help="skip the checkerboard contact sheet")

    s = p.add_argument_group("server")
    s.add_argument("--serve", action="store_true", help="run the browser UI instead of converting")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--keep-work", action="store_true", help="keep the intermediate frame directory")
    return p


class _Progress:
    """Single-line stage/percentage reporter."""

    LABELS = {"probe": "Reading video", "decode": "Extracting frames", "model": "Loading model",
              "matte": "Matting frames", "analyse": "Smoothing matte", "compose": "Composing RGBA",
              "encode": "Encoding"}

    def __init__(self) -> None:
        self.stage = None
        self.t0 = time.time()

    def __call__(self, stage: str, done: int, total: int) -> None:
        if stage != self.stage:
            if self.stage:
                sys.stderr.write("\n")
            self.stage = stage
        pct = 100 * done / total if total else 100
        sys.stderr.write(f"\r  {self.LABELS.get(stage, stage):<18} {pct:5.1f}%  ")
        sys.stderr.flush()

    def done(self) -> None:
        sys.stderr.write(f"\r{' ' * 40}\r")
        sys.stderr.flush()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.serve:
        from .web import run_server
        run_server(host=args.host, port=args.port)
        return 0

    if args.input is None:
        build_parser().error("an input video is required (or use --serve)")

    opts = StickerOptions(
        fps=args.fps, size=args.size, work_size=args.work_size, crop=args.crop,
        pad=args.pad, square=args.square, smooth=args.smooth, formats=args.formats,
        start=args.start, duration=args.duration, loop=args.loop,
        alpha_threshold=args.alpha_threshold, dither=args.dither, colors=args.colors,
        webp_quality=args.webp_quality, preview=args.preview,
        matte=MatteConfig(
            mode=args.mode, model=args.model,
            key=tuple(args.key) if args.key else None,
            tolerance=args.tolerance, softness=args.softness, despill=args.despill,
            feather=args.feather, shrink=args.shrink,
        ),
    )

    reporter = None if args.quiet else _Progress()
    try:
        result = create_sticker(args.input, args.out, opts, progress=reporter,
                                stem=args.name, keep_work=args.keep_work)
    except Exception as exc:  # surfaced as a message, not a traceback
        if reporter:
            reporter.done()
        print(f"vidsticker: {exc}", file=sys.stderr)
        return 1

    if reporter:
        reporter.done()
    if not args.quiet:
        print(f"Matte: {result.mode}" + (f" (chroma key Cr/Cb {result.key[0]:.0f}/{result.key[1]:.0f})"
                                         if result.key else ""))
        print(f"Output: {result.frames} frames @ {result.fps:g} fps, "
              f"{result.width}x{result.height} px")
        for fmt, path in sorted(result.outputs.items()):
            print(f"  {fmt:<5} {path}  ({path.stat().st_size / 1e6:.2f} MB)")
        if result.preview:
            print(f"  sheet {result.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
