# CORTEX Dashboard — Accessibility Notes

## Implemented

- **Skip link**: "Skip to main content" at top; visible on keyboard focus.
- **Landmarks**: `<main id="main">`, `<header role="banner">`, `<nav>`, `<aside aria-label="Main navigation">`.
- **Focus visible**: All interactive elements use `:focus-visible` (2px accent ring, 2px offset).
- **ARIA**: `aria-label` on icon-only buttons (sidebar toggle, search, notifications, voice, profile). `aria-current="page"` on active nav link. `aria-expanded` on collapse/expand and dropdowns. `aria-pressed` on voice button.
- **Keyboard**: Tab order follows layout (skip → sidebar → top bar → main). Sidebar and top bar buttons and links are focusable. Form inputs and buttons are keyboard operable.
- **Reduced motion**: `@media (prefers-reduced-motion: reduce)` in `index.css` shortens animations to 0.01ms so motion is effectively disabled.
- **Contrast**: Text/background ratios use design tokens (--text-primary on --bg-base) meeting WCAG AA for normal text. Optional high-contrast mode is available in Settings (data-contrast="high") for stronger borders and secondary text.

## Recommendations

1. **Focus trap in modals**: When adding modals, trap focus inside and restore focus on close.
2. **Live regions**: For dynamic alerts (toasts, notifications), use `aria-live="polite"` or `aria-live="assertive"` on a dedicated region.
3. **Chart accessibility**: Provide a text summary or table alternative for key chart data; ensure tooltips are keyboard-accessible (e.g. focusable trigger or visible on focus).
4. **Voice module**: Announce "Listening…" and "Stopped" via `aria-live` for screen reader users.
5. **Testing**: Run axe DevTools or Lighthouse accessibility audit; test with keyboard only and with a screen reader (NVDA, VoiceOver).

## Performance

- Route-based code splitting: use `React.lazy()` for page components if the bundle grows.
- Charts: Recharts is lazy-friendly; consider dynamic import for the chart component.
- Animations: Prefer `transform` and `opacity`; avoid animating `width`/`height` where possible to reduce layout thrash.
