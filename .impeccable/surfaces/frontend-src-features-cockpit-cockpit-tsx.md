---
version: 1
slug: "frontend-src-features-cockpit-cockpit-tsx"
primary_target: "frontend/src/features/cockpit/Cockpit.tsx"
related_targets: ["frontend/src/features"]
---

# Fine-tuning cockpit

Scope: the studio's surface for making and improving one fine-tuned model (a project). Visitor mode: Operate. Both Business and Research modes use it; Research gets full settings, logs and every number in the inspector.

Audience and job: a business user turns their knowledge, examples or agent traffic into a proven model they can deploy; a researcher iterates versions on evidence. Success: from a sentence to an evaluated fine-tune in one sitting, then each next version justified by the scorecard.

Constraints: the opencontroller console visual system stays (tokens, shadcn primitives, light and dark); embeddable in opencontroller; every action Ctrl agent takes is visible, reversible where possible, and has an SDK/CLI equivalent; no invented metrics.

## Direction contract

THESIS: Ctrl agent and the loop are one instrument: a conversation proposes each next step as an approvable action, and a live map of the whole loop (data, fine-tune, evaluate, deploy) shows the consequence the moment you approve. It refuses the category default of a settings form followed by a separate runs page.

OWN-WORLD: The console's Paper & Ink: white canvas, warm hairlines, slate-blue tint for what can be acted on and what is live, earthy good/warning/destructive tints for verdicts. Action cards are bordered panels with one tinted primary and one quiet edit. Loop nodes are four hairline-framed stations on a single rail, the active one tinted, the rail drawn as one continuous line whose segment flows while work runs.

STORY: The visitor says what the model should do; Ctrl agent answers with a plan they approve; the map lights station by station; the scorecard lands in the Evaluate station and Ctrl agent proposes the next version or the deploy. They leave knowing whether their model is ready and why.

FIRST VIEWPORT: Left 7/12: the loop rail across the top (Data → Fine-tune → Evaluate → Deploy, each station with its status line and version chips), the selected station's inspector filling the rest (examples, live loss curve, scorecard, endpoint). Right 5/12: Ctrl agent's conversation, header with the project's name and stage, thread of Ctrl agent's messages and action cards, composer pinned at the bottom with suggested next actions as chips (stacked on a phone, Ctrl agent comes first). The primary action is always the newest action card's tinted button in the conversation.

FORM: Copilot cockpit (the agent named Ctrl agent, placed on the right by the user after the build), position 1 of 7 in the ranked list (dealt second); seed key f7febb6b. Signature interaction: approving an action card lights its station and runs the rail segment into it; hovering a message highlights its station; clicking a station scrolls the thread to its latest message. Motion grammar: 150–250 ms ease-out state changes, a dash-flow on the active rail segment while a job runs, all of it off under reduced motion.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Unresolved

- Free-text understanding is rule-based (intents and goal classification) in this build; planning with the user's frontier model is a follow-up.
