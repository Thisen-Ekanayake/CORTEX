# CORTEX Dashboard — Quick Style Guide

Quick reference for designers and developers. Full spec: [DESIGN_SPEC.md](./DESIGN_SPEC.md).

## Colors (Dark)

| Use | Token | Value |
|-----|-------|--------|
| Background | `--bg-base` | #0d1117 |
| Cards / panels | `--surface-glass` | rgba(22,27,34,0.6) |
| Text primary | `--text-primary` | #e6edf3 |
| Text muted | `--text-secondary` | #8b949e |
| Primary action | `--accent-primary` | #58a6ff |
| Success | `--accent-success` | #3fb950 |
| Warning | `--accent-warning` | #d29922 |
| Error | `--accent-error` | #f85149 |

## Typography

- **Scale**: 0.75rem → 2.5rem (xs to display).
- **Weights**: 400, 500, 600, 700.
- **Font**: Inter, SF Pro Display, system-ui.

## Spacing

- Base: 4px. Scale: 4, 8, 12, 16, 24, 32, 48, 64.

## Components

- **Sidebar**: 260px / 72px collapsed; glass + blur 16px.
- **Top bar**: 56px; search, notifications, profile.
- **Cards**: 12px radius, 20px padding, glass + blur, hover lift.
- **Buttons**: Primary (accent), Secondary (border), Ghost (transparent); 8px radius.

## Motion

- Fast: 150ms. Normal: 250ms. Slow: 400ms.
- Easing: `cubic-bezier(0.4, 0, 0.2, 1)`.
- Respect `prefers-reduced-motion: reduce`.
