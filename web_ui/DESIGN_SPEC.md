# CORTEX Dashboard — UI Design Specification

A feature-rich interactive dashboard for the CORTEX RAG/AI assistant. Dark theme, glassmorphism, micro-animations, and full accessibility.

---

## 1. Design Philosophy

- **Dark-first**: Soft contrast, nuanced blacks/dark grays/dark blues (no pure `#000`).
- **Depth through translucency**: Glassmorphism and layered panels separate content without clutter.
- **Kinetic feedback**: Every interactive element has micro-animations for polish and feedback.
- **Accessible by default**: WCAG 2.1 AA, keyboard navigation, ARIA, contrast options.

---

## 2. Color Palette (Dark Theme)

### Semantic Tokens

| Token | Hex / Value | Usage |
|-------|-------------|--------|
| `--bg-base` | `#0d1117` | Page background |
| `--bg-elevated` | `#161b22` | Cards, panels |
| `--bg-overlay` | `rgba(22, 27, 34, 0.85)` | Glass overlay |
| `--surface-glass` | `rgba(22, 27, 34, 0.6)` | Glassmorphism panels |
| `--border-subtle` | `rgba(48, 54, 61, 0.6)` | Borders |
| `--text-primary` | `#e6edf3` | Headings, body |
| `--text-secondary` | `#8b949e` | Muted text |
| `--text-tertiary` | `#6e7681` | Placeholders |
| `--accent-primary` | `#58a6ff` | Links, primary actions |
| `--accent-success` | `#3fb950` | Success, positive metrics |
| `--accent-warning` | `#d29922` | Warnings, alerts |
| `--accent-error` | `#f85149` | Errors, destructive |
| `--gradient-subtle` | `linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1520 100%)` | Background gradient |

### Glassmorphism

- **Backdrop blur**: `12px`–`20px` (cards), `8px` (inputs).
- **Opacity**: Surface 60–85%, borders 40–60%.
- **Border**: `1px solid rgba(255,255,255,0.06)` for light edge.

---

## 3. Typography

### Scale (modular scale 1.25)

| Name | Size | Weight | Line height | Use |
|------|------|--------|-------------|-----|
| `--font-xs` | 0.75rem | 500 | 1.25 | Labels, badges |
| `--font-sm` | 0.875rem | 400 | 1.5 | Body small |
| `--font-base` | 1rem | 400 | 1.5 | Body |
| `--font-lg` | 1.125rem | 500 | 1.4 | Subheadings |
| `--font-xl` | 1.25rem | 600 | 1.3 | Section titles |
| `--font-2xl` | 1.5rem | 600 | 1.25 | Page titles |
| `--font-3xl` | 2rem | 700 | 1.2 | Hero / key metrics |
| `--font-display` | 2.5rem–3rem | 700 | 1.1 | Kinetic headings |

### Kinetic Typography

- Key metrics and hero numbers: subtle scale/opacity on scroll or on load (Framer Motion).
- Headings: optional staggered letter or word reveal (0.03–0.05s delay per unit).
- Font stack: `'Inter', 'SF Pro Display', system-ui, sans-serif`.

---

## 4. Spacing & Layout

- **Base unit**: `4px`.
- **Spacing scale**: `4, 8, 12, 16, 24, 32, 48, 64, 96`.
- **Sidebar width**: Expanded `260px`, collapsed `72px`.
- **Top bar height**: `56px`.
- **Content max-width**: `1400px`; padding `24px` (desktop), `16px` (tablet/mobile).
- **Grid**: 12-column; cards use `grid-template-columns: repeat(auto-fill, minmax(280px, 1fr))` where appropriate.

---

## 5. Component Library (Figma/Sketch Reference)

### 5.1 Navigation

- **Sidebar**
  - Background: `--surface-glass`, backdrop-blur `16px`.
  - Items: icon (24px) + label; padding 12px 16px; border-radius 8px.
  - Hover: `background: rgba(255,255,255,0.06)`; transition 200ms ease.
  - Active: `background: rgba(88,166,255,0.12)`; left border 3px `--accent-primary`.
  - Collapsed: icons only, tooltip on hover (aria-describedby).

- **Top bar**
  - Height 56px; same glass style as sidebar.
  - Left: menu (collapse) + breadcrumb or page title.
  - Center: global search (glass input, 320px max-width).
  - Right: context menu trigger, notifications (badge), profile avatar.

### 5.2 Cards

- **Default card**
  - Background: `--surface-glass`, blur 16px, border `--border-subtle`.
  - Border-radius: `12px`; padding: `20px`; box-shadow: `0 4px 24px rgba(0,0,0,0.2)`.
  - Hover: slight scale (1.01), shadow increase; transition 250ms cubic-bezier(0.4, 0, 0.2, 1).

- **Stat card**
  - Same as default; add large number (--font-3xl) with optional count-up animation.
  - Subtitle and optional sparkline or mini chart.

### 5.3 Buttons

- **Primary**: `--accent-primary` background; hover brighten 10%; active scale 0.98.
- **Secondary**: transparent + border; hover `rgba(255,255,255,0.08)`.
- **Ghost**: transparent; hover `rgba(255,255,255,0.06)`.
- All: padding 10px 20px; border-radius 8px; font-weight 500; transition 200ms.

### 5.4 Form Controls

- **Input**: glass background, border, 8px radius; focus ring 2px `--accent-primary` offset 2px.
- **Toggle**: track 40×24px; thumb 20px; colors: track off `--bg-elevated`, on `--accent-primary`.
- **Checkbox / Radio**: 20px; custom with focus-visible ring.

### 5.5 Charts

- **Library**: Recharts (or Chart.js) with custom dark theme.
- **Colors**: Sequential palette from `--accent-primary` with opacity steps.
- **Tooltips**: Glass panel, 8px radius, padding 12px; enter/exit 150ms ease.
- **Animations**: Data series animate on mount (duration 600–800ms, easing ease-out).

### 5.6 Chat / Assist Panel

- **Message bubble**: User right (accent tint), assistant left (glass).
- **Typed animation**: Character-by-character or word-by-word (configurable).
- **Smooth scroll**: New messages trigger scroll-into-view with smooth behavior.
- **Input**: Sticky bottom; send button with hover/active states.

### 5.7 Modals & Overlays

- **Backdrop**: `rgba(0,0,0,0.6)`; blur 8px.
- **Modal**: Centered; glass panel; max-width 480px; scale 0.95→1, opacity 0→1 (200ms).

---

## 6. Motion & Animation

### Timing

- **Instant**: 100ms (toggle, cursor).
- **Fast**: 200ms (buttons, hover).
- **Normal**: 300ms (cards, panels).
- **Slow**: 500ms (page transitions, charts).
- **Easing**: `cubic-bezier(0.4, 0, 0.2, 1)` default; ease-out for leave.

### Micro-interactions

- Buttons: scale 0.98 on click; hover lift (translateY -1px) optional.
- Cards: hover scale 1.01, shadow 0 8px 32px.
- Loading: skeleton pulse (opacity 0.4–0.8, 1.5s infinite) or spinner (rotate 1s linear).
- Toasts: slide in from top-right; auto-dismiss with progress bar.

### Parallax

- Optional subtle parallax (0.02–0.05) on hero or background layers; reduce or disable on `prefers-reduced-motion`.

---

## 7. Responsive Breakpoints

| Breakpoint | Width | Behavior |
|------------|--------|----------|
| `sm` | 640px | Sidebar overlay on mobile; single column cards. |
| `md` | 768px | Two-column where applicable. |
| `lg` | 1024px | Sidebar visible; 3-column grid. |
| `xl` | 1280px | Full layout; max content width. |

- Mobile: sidebar drawer (overlay); top bar full width; search collapses to icon opening full-screen search.

---

## 8. Accessibility

- **Focus**: All interactive elements have visible focus ring (2px `--accent-primary`, offset 2px).
- **Skip link**: “Skip to main content” at top; visible on focus.
- **ARIA**: Landmarks (banner, nav, main, complementary), live regions for alerts/notifications, aria-labels on icon-only buttons.
- **Keyboard**: Tab order logical; sidebar and modals trap focus; Escape closes overlays.
- **Contrast**: Text/background ≥ 4.5:1 (AA); large text ≥ 3:1. Optional high-contrast mode via CSS class or preference.
- **Reduced motion**: Respect `prefers-reduced-motion: reduce` (disable parallax, shorten/simplify animations).

---

## 9. Optional: Voice Command Module

- **Trigger**: Mic button in top bar or search.
- **Feedback**: Pulsing ring while listening; transcript shown in search or dedicated strip.
- **Commands**: “Go to Home”, “Search for …”, “Open documents”, “Toggle dark mode”.
- **Fallback**: Clear message if not supported; use Web Speech API with graceful degradation.

---

## 10. Performance

- **Code splitting**: Route-based lazy loading for pages.
- **Images**: Placeholder blur or skeleton; responsive srcset if applicable.
- **Charts**: Lazy load Recharts; virtualize long lists (e.g. document browser).
- **Animations**: Prefer `transform` and `opacity`; avoid layout thrashing.

---

## 11. File Structure (Implementation)

```
web_ui/
├── DESIGN_SPEC.md           # This document
├── PUBLIC_STYLE_GUIDE.md    # Quick reference (optional)
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css            # Theme variables, base styles
│   ├── theme/
│   │   └── tokens.css       # Design tokens
│   ├── components/
│   │   ├── layout/          # Sidebar, TopBar, Layout
│   │   ├── ui/               # Button, Card, Input, Toggle, etc.
│   │   ├── charts/           # Wrapped Recharts + tooltips
│   │   └── voice/            # VoiceCommand module
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── Search.tsx
│   │   ├── Documents.tsx
│   │   ├── Chat.tsx
│   │   └── Settings.tsx
│   ├── hooks/                # useReducedMotion, useTheme, etc.
│   └── data/                 # Placeholder data
└── README.md
```

---

*Document version: 1.0 — CORTEX Dashboard Design Specification*
