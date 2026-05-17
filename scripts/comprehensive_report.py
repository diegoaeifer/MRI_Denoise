"""
Extended HTML report for the comprehensive MRI denoising benchmark.

Adds HaarPSI/VGG loss charts and dataset×modality breakdown table
on top of the standard benchmark_report layout.

Usage
-----
    from scripts.comprehensive_report import generate_comprehensive_report
    html_path = generate_comprehensive_report(csv_path, output_dir)
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _fig_to_b64(fig) -> str:
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return b64


def _metric_boxplot(df, metric: str, title: str, ylabel: str) -> str:
    """Return base64 PNG boxplot for *metric* grouped by model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    data = [df.loc[df["model"] == m, metric].dropna().values for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
    ax.boxplot(data, tick_labels=models, patch_artist=True)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _leaderboard_html(df) -> str:
    """HTML table with full leaderboard (PSNR, SSIM, HaarPSI, VGG, Sharpness)."""
    noisy_mask = df["model"] == "noisy_input"
    cols = ["psnr", "ssim", "haarpsi", "vgg_loss", "sharpness"]
    avail = [c for c in cols if c in df.columns]
    lb = (
        df[~noisy_mask]
        .groupby("model")[avail]
        .mean()
        .sort_values("psnr", ascending=False)
        .reset_index()
    )

    header = "<tr><th>Rank</th><th>Model</th><th>PSNR (dB) ↑</th><th>SSIM ↑</th>"
    if "haarpsi" in avail:
        header += "<th>HaarPSI ↑</th>"
    if "vgg_loss" in avail:
        header += "<th>VGG Loss ↓</th>"
    if "sharpness" in avail:
        header += "<th>Sharpness ↑</th>"
    header += "</tr>\n"

    rows = ""
    for rank, row in enumerate(lb.itertuples(index=False), 1):
        def fmt(v):
            return f"{v:.4f}" if v == v else "N/A"

        rows += (f"<tr><td>{rank}</td><td>{row.model}</td>"
                 f"<td>{fmt(row.psnr)}</td><td>{fmt(row.ssim)}</td>")
        if "haarpsi" in avail:
            rows += f"<td>{fmt(row.haarpsi)}</td>"
        if "vgg_loss" in avail:
            rows += f"<td>{fmt(row.vgg_loss)}</td>"
        if "sharpness" in avail:
            rows += f"<td>{fmt(row.sharpness)}</td>"
        rows += "</tr>\n"

    return (
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;width:100%;'>"
        f"<thead>{header}</thead><tbody>{rows}</tbody></table>"
    )


def _dataset_modality_table_html(df) -> str:
    """Mean PSNR per (model, dataset, modality) pivot table."""
    noisy_mask = df["model"] == "noisy_input"
    sub = df[~noisy_mask]
    if sub.empty or "dataset" not in sub.columns or "modality" not in sub.columns:
        return "<p>No data for breakdown table.</p>"
    pivot = (
        sub.groupby(["model", "dataset", "modality"])["psnr"]
        .mean()
        .reset_index()
        .pivot_table(
            index="model",
            columns=["dataset", "modality"],
            values="psnr",
            aggfunc="mean",
        )
    )
    if pivot.empty:
        return "<p>No data for breakdown table.</p>"
    return pivot.round(2).to_html(border=1, justify="center")


def generate_comprehensive_report(csv_path: Path, output_dir: Path) -> Path:
    """Generate an extended HTML benchmark report.

    Parameters
    ----------
    csv_path : Path
        ``benchmark_results.csv`` produced by comprehensive_benchmark.py.
    output_dir : Path
        Directory where ``report_comprehensive.html`` is written.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df_2d = df[df["track"] == "2d"] if "track" in df.columns else df
    noisy_mask = df_2d["model"] == "noisy_input"
    df_models = df_2d[~noisy_mask]

    sections = []

    sections.append(
        "<h2>Leaderboard — Track A 2D "
        "(mean across all datasets × modalities × σ)</h2>"
    )
    sections.append(_leaderboard_html(df_2d))

    if not df_models.empty:
        sections.append("<h2>PSNR Distribution</h2>")
        b64 = _metric_boxplot(df_models, "psnr", "PSNR per model", "PSNR (dB)")
        sections.append(
            f'<div class="figure"><img src="data:image/png;base64,{b64}"'
            f' style="max-width:100%;"></div>'
        )

        sections.append("<h2>SSIM Distribution</h2>")
        b64 = _metric_boxplot(df_models, "ssim", "SSIM per model", "SSIM")
        sections.append(
            f'<div class="figure"><img src="data:image/png;base64,{b64}"'
            f' style="max-width:100%;"></div>'
        )

        if "haarpsi" in df_models.columns and df_models["haarpsi"].notna().any():
            sections.append("<h2>HaarPSI Distribution</h2>")
            b64 = _metric_boxplot(
                df_models, "haarpsi", "HaarPSI per model", "HaarPSI"
            )
            sections.append(
                f'<div class="figure"><img src="data:image/png;base64,{b64}"'
                f' style="max-width:100%;"></div>'
            )

        if "vgg_loss" in df_models.columns and df_models["vgg_loss"].notna().any():
            sections.append("<h2>VGG Perceptual Loss (lower = better)</h2>")
            b64 = _metric_boxplot(
                df_models, "vgg_loss", "VGG Loss per model", "VGG Loss"
            )
            sections.append(
                f'<div class="figure"><img src="data:image/png;base64,{b64}"'
                f' style="max-width:100%;"></div>'
            )

    sections.append("<h2>Mean PSNR by Dataset × Modality</h2>")
    sections.append(_dataset_modality_table_html(df_2d))

    if "track" in df.columns:
        df_3d = df[df["track"] == "3d"]
        if not df_3d.empty:
            sections.append("<h2>Leaderboard — Track B 3D</h2>")
            sections.append(_leaderboard_html(df_3d))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Comprehensive MRI Denoising Benchmark</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1, h2 {{ color: #333; }}
    table {{ margin-bottom: 30px; border-collapse: collapse; }}
    th {{ background: #ddeeff; padding: 6px; }}
    td {{ text-align: center; padding: 6px; }}
    .figure {{ margin: 20px 0; }}
  </style>
</head>
<body>
  <h1>Comprehensive MRI Denoising Benchmark Report</h1>
  <p>Source: <code>{csv_path}</code></p>
  {"".join(sections)}
</body>
</html>"""

    html_path = output_dir / "report_comprehensive.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Report written: {html_path}")
    return html_path
