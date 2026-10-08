---
name: openfinetuner
description: The fine-tuning studio (formerly ShadowLM), in the opencontroller console's Paper & Ink, softened.
colors:
  canvas: "#ffffff"
  ink-foreground: "#141311"
  surface: "#ede9e1"
  surface-2: "#e8e4dd"
  subtle: "#f6f3ed"
  side-panel: "#f8f7f4"
  muted-foreground: "#635d51"
  hairline: "#e3ded4"
  hairline-strong: "#d2cdc0"
  input-edge: "#e8e4dd"
  slate-primary: "#3d5a86"
  primary-foreground: "#fdfcf9"
  good: "#3f7a4f"
  warning: "#a5671c"
  destructive: "#9e3b28"
  console-ink: "#141311"
  console-bone: "#ebe8e3"
  dark-canvas: "#121110"
  dark-background: "#1a1917"
  dark-popover: "#1e1d1b"
  dark-surface: "#201f1c"
  dark-surface-2: "#262522"
  dark-foreground: "#ebe8e3"
  dark-muted-foreground: "#a39e95"
  dark-hairline: "#2a2926"
  dark-hairline-strong: "#38362f"
  dark-slate-primary: "#9db2d4"
  dark-good: "#6faa83"
  dark-warning: "#cfa25a"
  dark-destructive: "#d4685a"
typography:
  headline:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "-0.015em"
  title:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "-0.01em"
  figure:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 600
    lineHeight: 1.333
    letterSpacing: "-0.01em"
    fontFeature: "\"tnum\""
  body:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.625
    fontFeature: "\"cv11\", \"ss01\""
  nav:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 500
  label:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
  meta:
    fontFamily: "Manrope, ui-sans-serif, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 500
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, SF Mono, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.625
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
  xl: "12px"
  pill: "9999px"
spacing:
  hair: "2px"
  xs: "4px"
  sm: "8px"
  md: "12px"
  cell-y: "14px"
  lg: "16px"
  panel-x: "20px"
  page: "24px"
components:
  button-primary:
    backgroundColor: "color-mix(in srgb, {colors.slate-primary} 10%, transparent)"
    textColor: "{colors.slate-primary}"
    rounded: "{rounded.lg}"
    padding: "0 10px"
    height: "32px"
  button-primary-hover:
    backgroundColor: "color-mix(in srgb, {colors.slate-primary} 15%, transparent)"
  button-outline:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink-foreground}"
    rounded: "{rounded.lg}"
    padding: "0 10px"
    height: "32px"
  button-outline-hover:
    backgroundColor: "{colors.surface-2}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.muted-foreground}"
    rounded: "{rounded.lg}"
    padding: "0 10px"
    height: "32px"
  button-destructive:
    backgroundColor: "color-mix(in srgb, {colors.destructive} 10%, transparent)"
    textColor: "{colors.destructive}"
    rounded: "{rounded.lg}"
    height: "32px"
  input:
    backgroundColor: "transparent"
    textColor: "{colors.ink-foreground}"
    rounded: "{rounded.lg}"
    padding: "4px 10px"
    height: "32px"
  chip-suggestion:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink-foreground}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "0 8px"
    height: "24px"
  badge:
    backgroundColor: "color-mix(in srgb, {colors.slate-primary} 10%, transparent)"
    textColor: "{colors.slate-primary}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
    height: "20px"
  version-chip:
    backgroundColor: "color-mix(in srgb, {colors.slate-primary} 10%, transparent)"
    textColor: "{colors.slate-primary}"
    rounded: "{rounded.pill}"
    padding: "0 6px"
    height: "20px"
  panel:
    backgroundColor: "{colors.canvas}"
    rounded: "{rounded.lg}"
    padding: "14px 16px"
  nav-item:
    backgroundColor: "transparent"
    textColor: "{colors.muted-foreground}"
    typography: "{typography.nav}"
    rounded: "{rounded.md}"
    padding: "0 6px"
    height: "32px"
  nav-item-active:
    backgroundColor: "color-mix(in srgb, {colors.slate-primary} 10%, transparent)"
    textColor: "{colors.slate-primary}"
  station-marker:
    backgroundColor: "{colors.canvas}"
    rounded: "{rounded.pill}"
    size: "24px"
  log-console:
    backgroundColor: "{colors.console-ink}"
    textColor: "{colors.console-bone}"
    typography: "{typography.mono}"
    padding: "16px"
---

# Design System: openfinetuner

## Overview

**Creative North Star: "The Lab Notebook on Warm Paper"**

openfinetuner wears the opencontroller console's system, Paper & Ink, softened, on purpose: the studio embeds inside opencontroller and has to read as the same instrument. Near-black ink sits on a white page; every neutral is warmed toward paper (no pure black, no cool greys); panels are white blocks set apart by a warm hairline rather than by depth. Corners are near-square and soften only as surfaces grow.

The density is a working console's: small type (14px body, 12px labels, 11px meta), tight but regular padding, and every number in tabular figures. Nothing is filled solid. The single interactive colour, a muted slate blue, marks what can be acted on and where you are, always as a soft tint with deeper text. Verdicts speak in earthy tints the same way: moss green for good, ochre for not-there-yet, brick for failure. Data, ids, model names and code are set in JetBrains Mono, so what is machine-truth reads differently from what is prose.

The cockpit added the system's first signature components: a Ctrl agent thread whose action cards carry one tinted primary and a quiet Edit, a four-station loop rail drawn as one continuous line, a station inspector, and a scorecard. Light and dark are equal citizens: dark is warm graphite with the same roles, and the embed host's theme wins when framed.

**Key Characteristics:**
- White canvas, warm hairlines, warm-brown faint shadows; depth by edge, not elevation.
- One slate-blue accent, used only as a tint (10% fill, 20-30% edge, full-strength text).
- Earthy verdict tints (good / warning / destructive) in the same tint grammar.
- Manrope for every word, JetBrains Mono for every datum; tabular numbers throughout.
- Near-square corners from one radius (8px); pills only for pills.
- Motion is short and functional (150-250ms ease-out) and disappears under reduced motion.

## Colors

A warm-neutral paper palette with one slate-blue voice and three earthy verdict tints, mirrored into warm graphite for dark.

### Primary
- **Slate Primary** (`slate-primary`; dark `dark-slate-primary`): the one interactive colour. Primary buttons, the active nav row, the selected loop station, ready/running station words, version chips, the "Open in Research" link, focus rings, checkbox accent, text selection (24% mix) and the caret. Always a tint: background at 5-15%, edge at 15-40%, the token itself only for text, icons, the loss curve and the running dot.

### Secondary
- **Moss Good** (`good`): passed verdicts, done stations, correct-answer checks, the eval point on the loss curve, the "live" dot. Aliased as `success` in code.
- **Ochre Warning** (`warning`): "not there yet" verdicts and the Evaluate station when the fine-tune beats its base but misses the bar.
- **Brick Destructive** (`destructive`): failures, wrong-answer crosses, destructive actions, validation errors.

### Neutral
- **Canvas** (`canvas`): the page, cards and panels in light mode; content reads crisp on white.
- **Ink Foreground** (`ink-foreground`): all body text and headings; warm near-black, never #000.
- **Muted Foreground** (`muted-foreground`): secondary lines, labels, meta, idle stations.
- **Surface / Surface 2** (`surface`, `surface-2`): raised fills: table heads, code backgrounds (often at 40%), segmented-control track, outline-button hover.
- **Subtle** (`subtle`): a whisper for hover rows, zebra rows and the person's own messages in the thread.
- **Side Panel** (`side-panel`): the floating nav island; the warmth lives here, in the fills and in the rules.
- **Hairline / Hairline Strong** (`hairline`, `hairline-strong`): dividers and panel edges; the stronger edge for markers in a muted state and thin scrollbars.
- **Console Ink / Bone** (`console-ink`, `console-bone`): the training-log console, dark in either theme; also the phone overlay scrim (ink at 30%).

### Named Rules
**The Tint, Never Fill Rule.** No component fills solid with an accent or verdict colour. Primary, good, warning and destructive appear as a 5-15% background with a 20-40% edge and full-strength text. A solid slate button is off-system.

**The Warm Neutral Rule.** Every grey leans toward paper. No pure black text or surface, no cool grey, and in dark mode no neutral greys: graphite carries a hint of the same warmth.

**The Verdict Earns Its Colour Rule.** Good, warning and destructive appear only where a real state or score backs them; a decorative green or ochre is a lie on a surface that must not invent metrics.

## Typography

**Body Font:** Manrope (with ui-sans-serif, system-ui)
**Heading Font:** Manrope, the same face, so the page and side panel read as one voice
**Mono Font:** JetBrains Mono (with ui-monospace, SF Mono)

**Character:** a quiet geometric sans for every word, set small and tight, with a crisp mono for everything a machine produced. Hierarchy comes from size and a step of weight (500 to 600) plus slight negative tracking, never from heavy weights or capitals.

### Hierarchy
- **Headline** (600, 1.25rem, -0.015em, balanced wrap): page titles in the page header and the mode chooser.
- **Title** (600, 1rem, -0.01em): panel and station headers, the cockpit's project name, action-card titles (at 0.875rem).
- **Figure** (600, 1.5rem, tabular): scorecard scores and stat-strip values.
- **Body** (400, 0.875rem, relaxed leading; descriptions capped at 64-70ch): Ctrl agent messages, table cells, descriptions (page descriptions at 13px).
- **Nav** (500, 13px): side-panel rows.
- **Label** (500, 0.75rem): buttons at xs, field labels, chips, station state lines, captions.
- **Meta** (500, 11px): message author line, station status word, side-panel section titles, "formerly ShadowLM", version strings.
- **Mono** (400, 0.75rem; 11px in consoles and dense metadata; 10px in the monogram and version chips): run ids, model ids, config keys, CLI blocks, logs, chart axes. Ligatures off in typed mono fields.

### Named Rules
**The Mono Is Machine-Truth Rule.** Anything a person could paste into a shell or an API (ids, model names, config keys, commands, logs) is JetBrains Mono; prose never is.

**The No Capitals Rule.** Labels are sentence case at 11-12px, weight 500. Uppercase tracked labels are not part of this system.

## Layout

The studio shell is a floating side-panel island (240px open, 48px folded to an icon rail; 12px inset from the viewport, 24px gap to the page). Pages pad 24px on the sides and top, 32px at the bottom, and use container queries rather than viewport breakpoints, so the same page works framed inside opencontroller.

Panels set their own internal rhythm: headers at 20px horizontal by 14px vertical with a hairline beneath, cells and callouts at 16px by 14px, sibling panels 16px apart, page header 28px above content. Grids of figures (stat strips, scorecard targets) are joined by a 1px hairline gap on a hairline-coloured ground, not by separate boxes.

The cockpit is a 12-column split at 56rem of its own width: conversation 5/12 on the left, loop rail and inspector 7/12 on the right, each pane scrolling inside the viewport. Below that width the panes stack, flow at natural height and the page becomes the only scroller; the composer goes sticky at the bottom and earlier messages fold behind "Show N earlier". The loop rail runs four columns wide and two columns (breaking after Fine-tune) when narrow. Tables of answers become stacked blocks below the xl container width.

Below 768px the side panel never takes page width: it folds to the 48px rail and opens as an overlay above an ink scrim at 30% with a 1px blur.

### Named Rules
**The Container Not Viewport Rule.** Components respond to their container's width (container queries, ResizeObserver), because the studio is embedded at sizes the viewport cannot predict.

**The One Scroller Rule.** On a phone the page is the only scroller; nested scroll panes exist only when panes sit side by side.

## Elevation & Depth

The system is flat with a hairline edge. Panels are separated from the canvas by a 1px warm hairline; a faint warm-brown shadow is reserved for the few things that float or that the eye should land on first: the side-panel island, the active segment of the mode switch, and the live action card. Popovers, selects, tooltips and sheets take the raised shadow. In dark mode both shadows switch to black at higher opacity, and depth comes mostly from canvas (darkest) to card to hover (lighter).

### Shadow Vocabulary
- **Paper** (`box-shadow: 0 1px 0 rgb(28 24 18 / 0.03), 0 1px 2px rgb(28 24 18 / 0.05)`): the side-panel island, the selected mode segment, the live action card.
- **Raised** (`box-shadow: 0 1px 0 rgb(28 24 18 / 0.04), 0 2px 4px rgb(28 24 18 / 0.06), 0 8px 24px -12px rgb(28 24 18 / 0.1)`): popovers, menus, select content, tooltips, sheets.

### Named Rules
**The Hairline First Rule.** Separate with a 1px warm hairline before reaching for a shadow; a shadow marks a floating surface or the one live card, never a resting panel.

## Shapes

Every radius derives from one 8px base: 4px for tiny inline targets, 6px for compact controls, chips, nav rows and loop stations, 8px for buttons, inputs and any box drawn with a full border (applied globally), 12px for the floating side panel. Pills are reserved for true pills: badges, version chips, the station markers, progress bars and status dots. Borders are always 1px; dashed hairlines mark empty or spent states (a spent proposal, an empty chart grid). Tables framed by a border clip their rows to its corners.

## Components

### Buttons
Tinted and quiet; a button announces it can be acted on without shouting.
- **Shape:** gently squared (8px); 32px tall by default, 24px (xs) and 28px (sm) compact sizes at 6px.
- **Primary:** slate tint: 10% slate background, 20% slate edge, slate text, 14px weight 500; hover deepens to 15%.
- **Outline:** card background, hairline edge, ink text; hover fills with surface-2. Used for suggestion chips and secondary actions.
- **Ghost:** no edge or fill, muted text; hover fills muted. The quiet Edit and Copy buttons.
- **Destructive:** brick tint at 10%, hover 20%.
- **Link:** slate text, underline on hover (the "Open in Research" pattern).
- **Hover / Focus:** colour transitions; focus shows the ring edge plus a 3px slate ring at 50%; press nudges down 1px.

### Chips and Badges
- **Suggestion chips:** outline xs buttons (24px, weight 400) above the composer; wrap on wide, one horizontally scrolling row when stacked.
- **Badges:** 20px pills in the tint grammar (primary, secondary surface, destructive, outline). Run status badges use good/warning/destructive tints at 10% with a 30% edge.
- **Version chips:** 20px mono pills on the Fine-tune station; the current version tinted slate, others hairline and muted.

### Cards / Containers
- **Corner Style:** 8px for bordered panels.
- **Background:** canvas (card), with surface at 40% for neutral callouts and code bodies.
- **Shadow Strategy:** none at rest (see Elevation & Depth).
- **Border:** 1px hairline.
- **Internal Padding:** 16px by 14px for callouts and cells; 20px for station and thread content.

### Inputs / Fields
- **Style:** 1px input-edge stroke, transparent fill, 8px radius, 32px tall (composer textarea grows from 36px to 128px). Bare native controls get the same treatment globally.
- **Focus:** edge turns slate, 3px slate ring at 50%.
- **Error / Disabled:** brick edge with a 20% brick ring; disabled at 50% opacity. Field labels sit above at 12px muted.

### Navigation
- **Side-panel island:** floating, side-panel fill, 12px radius, paper shadow. Brand row is the "of" mono monogram (28px, slate tint, 6px radius) with "openfinetuner" over "formerly ShadowLM" at 11px.
- **Rows:** 32px, 13px weight 500, muted with a 70%-opacity 16px line icon; hover fills surface-2 at 70%; active is the slate tint with full-opacity icon.
- **Section titles:** 11px muted labels; folded, they become a centred hairline.
- **Mode switch:** a two-segment radio (Business / Research) on a surface-2 track; the selected segment is card-white with the paper shadow. Folded, it becomes a single icon button.
- **Phone:** the island folds to the 48px rail and opens as an overlay over the ink scrim.

### Ctrl Agent Thread (signature)
- **Messages:** an 11px meta author line ("Ctrl agent" with a 12px line icon) and, at its right, the station it speaks about as a small link; text at 14px relaxed. The person's own messages are right-aligned blocks on subtle with a hairline, max 85% wide.
- **Linking:** hovering a message tints its station on the rail; new messages rise in (5px, 180ms).
- **Action card:** the one live proposal is a bordered panel with the paper shadow: a 14px semibold title, a two-column definition list (muted terms), "why" lines with a middle-dot, one tinted primary action and a ghost Edit, and a "Same run from your shell" disclosure revealing the CLI in mono on surface. Spent proposals collapse to a dashed-hairline line.
- **Live line:** while work runs, a strip at 5% slate above the composer with a flowing dash.

### Loop Rail (signature)
- Four stations (Data, Fine-tune, Evaluate, Deploy) on one continuous 2px line. Each marker (24px circle, card fill) sits alone on the line; the station's name (14px semibold), status word (11px, toned) and line (12px muted) hang below.
- **Segment states:** hairline when ahead, slate at 40% after a done station, moss at 60% between done stations, a slate dash flowing (600ms linear loop) into a running station.
- **Station states:** idle dashed-circle muted, ready slate dot, running pinging slate dot, done moss check, Evaluate verdicts ochre alert or muted minus, failed brick cross.
- **Selection:** selected station takes a 5% slate fill with a 30% edge; the hovered-from-thread station a 3% fill with a 15% edge. Arrow keys move between stations.

### Station Inspector and Scorecard (signature)
- **Station header:** 16px semibold title, 14px muted state line, an inline 12px "Open in Research" slate text link beneath, actions at the right; hairline below.
- **Scorecard:** a verdict callout in its tint (5% fill, 30% edge, toned semibold title, muted detail); per-target figures joined by hairline gaps (24px tabular score, "of N correct", a 6px pill progress bar in slate for the fine-tune and muted for comparisons, the model id in 11px mono); answers as a fixed table when wide, stacked blocks when narrow, each answer prefixed by a moss check or brick cross.
- **Log console:** ink background, bone text at 90%, 11px mono, max 288px tall with a thin scrollbar.

## Do's and Don'ts

### Do:
- **Do** express every accent and verdict as a tint: 5-15% fill, 15-40% edge, full-strength text.
- **Do** set ids, model names, config keys, commands and logs in JetBrains Mono, and every number in tabular figures.
- **Do** separate panels with a 1px warm hairline and keep the paper shadow for floating surfaces and the live action card.
- **Do** derive radii from the 8px base and keep pills for badges, chips, markers and bars.
- **Do** size layouts from the container (container queries at 56rem for the cockpit split), so embedded and phone widths behave.
- **Do** keep motion to 150-250ms ease-out state changes, plus the dash-flow only while work runs, all off under reduced motion.
- **Do** keep light and dark equal: every colour role has its warm-graphite counterpart.

### Don't:
- **Don't** fill a button, badge or callout solid with slate or a verdict colour.
- **Don't** use pure black (#000) for text or surfaces, or cool greys, in either theme.
- **Don't** colour a state good, warning or destructive without a real score or status behind it.
- **Don't** add uppercase tracked labels, kickers or eyebrows above titles.
- **Don't** introduce a second display face; Manrope carries headings.
- **Don't** add heavy or black shadows in light mode, or lift resting panels with elevation.
- **Don't** give the side panel page width on a phone, or nest scroll panes when the cockpit is stacked.
