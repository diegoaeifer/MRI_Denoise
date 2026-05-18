"""
HTML report generator for benchmark_pretrained.py results.

Reads benchmark_results.csv (produced by benchmark_pretrained.py), creates
matplotlib figures (PSNR boxplot, SSIM boxplot, sharpness bar chart), embeds
them as base64 PNGs in a self-contained HTML report, and writes a leaderboard
table.

Usage
-----
    python scripts/benchmark_report.py \\
        --csv  C:/projetos/MRI_Denoise/artifacts/benchmark/benchmark_results.csv \\
        --output C:/projetos/MRI_Denoise/artifacts/benchmark/report
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Encode a matplotlib Figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return encoded


def _img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;">'


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _psnr_boxplot(df) -> str:
    """Return base64 PNG of per-model PSNR boxplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    data = [df.loc[df["model"] == m, "psnr"].dropna().values for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    ax.boxplot(data, labels=models, vert=True, patch_artist=True)
    ax.set_xlabel("Model")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR distribution per model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _ssim_boxplot(df) -> str:
    """Return base64 PNG of per-model SSIM boxplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    data = [df.loc[df["model"] == m, "ssim"].dropna().values for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    ax.boxplot(data, labels=models, vert=True, patch_artist=True)
    ax.set_xlabel("Model")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM distribution per model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _sharpness_bar(df) -> str:
    """Return base64 PNG of per-model mean sharpness bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = (
        df.groupby("model")["sharpness"]
        .mean()
        .sort_values(ascending=False)
    )
    models = list(summary.index)
    values = list(summary.values)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    x = np.arange(len(models))
    ax.bar(x, values, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean sharpness (Laplacian variance)")
    ax.set_title("Sharpness per model (higher = sharper)")
    plt.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _leaderboard_html(df) -> str:
    """Return an HTML <table> with the per-model leaderboard."""
    noisy_mask = df["model"] == "noisy_input"
    leaderboard = (
        df[~noisy_mask]
        .groupby("model")[["psnr", "ssim", "sharpness"]]
        .mean()
        .sort_values("psnr", ascending=False)
        .reset_index()
    )

    rows = ""
    for rank, row in enumerate(leaderboard.itertuples(index=False), start=1):
        psnr_val = f"{row.psnr:.2f}" if row.psnr == row.psnr else "N/A"  # NaN check
        ssim_val = f"{row.ssim:.4f}" if row.ssim == row.ssim else "N/A"
        sharp_val = f"{row.sharpness:.6f}" if row.sharpness == row.sharpness else "N/A"
        rows += (
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td>{row.model}</td>"
            f"<td>{psnr_val}</td>"
            f"<td>{ssim_val}</td>"
            f"<td>{sharp_val}</td>"
            f"</tr>\n"
        )

    return f"""
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;">
  <thead>
    <tr>
      <th>Rank</th><th>Model</th>
      <th>Mean PSNR (dB)</th><th>Mean SSIM</th><th>Mean Sharpness</th>
    </tr>
  </thead>
  <tbody>
{rows}  </tbody>
</table>
"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MRI Denoising Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1, h2 {{ color: #333; }}
    table {{ margin-bottom: 30px; }}
    th {{ background: #ddeeff; }}
    td {{ text-align: center; }}
    .figure {{ margin: 20px 0; }}
  </style>
</head>
<body>
  <h1>MRI Denoising Benchmark Report</h1>
  <p>Generated from: <code>{csv_path}</code></p>

  <h2>Leaderboard (ranked by mean PSNR)</h2>
  {leaderboard}

  <h2>PSNR Distribution</h2>
  <div class="figure">{psnr_fig}</div>

  <h2>SSIM Distribution</h2>
  <div class="figure">{ssim_fig}</div>

  <h2>Sharpness (Laplacian Variance)</h2>
  <div class="figure">{sharp_fig}</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public entry point (importable by tests)
# ---------------------------------------------------------------------------

def generate_report(csv_path: Path, output_dir: Path) -> Path:
    """Generate an HTML benchmark report from *csv_path*.

    Parameters
    ----------
    csv_path : Path
        Path to ``benchmark_results.csv`` produced by benchmark_pretrained.py.
        Required columns: model, psnr, ssim, sharpness.
    output_dir : Path
        Directory where the report and figure PNGs are written.

    Returns
    -------
    Path
        Absolute path to the generated ``report.html`` file.
    """
    import pandas as pd  # lazy import

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = {"model", "psnr", "ssim", "sharpness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    log.info("Building figures...")
    psnr_b64 = _psnr_boxplot(df)
    ssim_b64 = _ssim_boxplot(df)
    sharp_b64 = _sharpness_bar(df)

    # Also save figures as standalone PNGs for convenience
    for name, b64 in [("psnr_boxplot", psnr_b64),
                      ("ssim_boxplot", ssim_b64),
                      ("sharpness_bar", sharp_b64)]:
        png_path = output_dir / f"{name}.png"
        png_path.write_bytes(base64.b64decode(b64))
        log.info("Saved figure: %s", png_path)

    leaderboard_html = _leaderboard_html(df)

    html = _HTML_TEMPLATE.format(
        csv_path=csv_path,
        leaderboard=leaderboard_html,
        psnr_fig=_img_tag(psnr_b64, "PSNR boxplot"),
        ssim_fig=_img_tag(ssim_b64, "SSIM boxplot"),
        sharp_fig=_img_tag(sharp_b64, "Sharpness bar chart"),
    )

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    log.info("Saved report: %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate HTML report from benchmark_pretrained.py CSV output."
    )
    p.add_argument(
        "--csv", type=Path, required=True,
        help="Path to benchmark_results.csv.",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("C:/projetos/MRI_Denoise/artifacts/benchmark/report"),
        help="Output directory for report.html and figure PNGs.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    generate_report(csv_path=args.csv, output_dir=args.output)


if __name__ == "__main__":
    main()
