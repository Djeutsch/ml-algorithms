"""Wrap dashboard/page.html into a standalone dashboard/index.html that runs on
any static web server.

Usage:  python scripts/build_standalone_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"

page = (DASH / "page.html").read_text()
sources = {p.stem: f"data/{p.name}" for p in sorted((DASH / "data").glob("*.csv"))}

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Eight ML Algorithms</title>
<link rel="icon" href="favicon.ico" sizes="any">
<link rel="apple-touch-icon" href="logo.png">
<meta name="description" content="Interactive dashboard: eight machine-learning algorithms run on sample data.">
<style>
  :root {
    --color-bg: #f7f7f5; --color-fg: #1b1d22; --color-fg-muted: #5b606c; --color-panel: #ffffff;
    --color-border-line: #e2e3e7; --color-ok: #1e7f4f; --color-warn: #9a5c00; --color-bad: #c43d2f;
    --cds-chart-grid: #eceef1; --cds-chart-axis: #c3c6cd; --cds-chart-reference: #8a8f9c;
    --cds-surface-popover: #ffffff; --cds-border: #d7d9de; --cds-text-secondary: #5b606c;
    --cds-shadow-popover: 0 6px 18px rgba(20, 22, 28, .12);
    --cds-font-size-caption: .8125rem; --cds-font-size-body: .9375rem; --cds-font-size-heading: 1.0625rem; --cds-font-size-title: 1.75rem;
    --cds-font-weight-medium: 600;
    --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 12px; --cds-gap-lg: 20px; --cds-gap-xl: 32px;
    --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 18px;
    --cds-radius: 8px; --cds-dur-slow: .3s; --cds-ease-out: cubic-bezier(.2, .7, .2, 1);
    --font-sans: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    --font-serif: "Source Serif 4", "Iowan Old Style", Georgia, serif;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --color-bg: #111318; --color-fg: #e7e9ee; --color-fg-muted: #9aa0ad; --color-panel: #181b21;
      --color-border-line: #2a2e37; --color-ok: #5cc28c; --color-warn: #e0a640; --color-bad: #ef6f5f;
      --cds-chart-grid: #23262e; --cds-chart-axis: #3a3f4a; --cds-chart-reference: #6b7180;
      --cds-surface-popover: #1f232b; --cds-border: #333844; --cds-text-secondary: #9aa0ad;
      --cds-shadow-popover: 0 6px 18px rgba(0, 0, 0, .45);
      color-scheme: dark;
    }
  }
  html, body { margin: 0; background: var(--color-bg); color: var(--color-fg); }
  body { font-family: var(--font-sans); font-size: var(--cds-font-size-body); line-height: 1.4; }
  #dash-root { padding-block: 24px 48px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
  .dash-skeleton { background: linear-gradient(90deg, var(--color-border-line), transparent 60%); background-size: 200% 100%; border-radius: var(--cds-radius); opacity: .5; }
  .site-nav { max-width: 80rem; margin: 0 auto 16px; font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); }
  .site-nav a { color: inherit; display: inline-flex; align-items: center; gap: 8px; }
  .site-nav img { width: 28px; height: 28px; }
</style>
<script src="vendor/d3.v7.9.0.min.js"></script>
<script>window.DASH_SOURCES = __SOURCES__;</script>
<script src="dash-runtime.js"></script>
</head>
<body>
<div id="dash-root">
<nav class="site-nav"><a href="../"><img src="logo.png" alt="">← Companion Jupyter Book</a></nav>
"""
FOOT = """
</div>
</body>
</html>
"""

out = HEAD.replace("__SOURCES__", json.dumps(sources)) + page + FOOT
(DASH / "index.html").write_text(out)
print(f"wrote dashboard/index.html ({len(out) / 1024:.0f} KB, {len(sources)} data files)")
