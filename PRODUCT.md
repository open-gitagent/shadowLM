# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Two audiences, served by two explicit modes of the same studio (confirmed):

- **Business users**: product, operations and AI leads at enterprises (Lyzr customers and teams like the ones Lyzr demos to). They want a model of their own for a job: one that knows their company's knowledge, does a specific task from their examples, or takes over a task their agent currently sends to a rented frontier model. They want it proven before they trust it, and they think in tasks, quality, cost and "can I ship it", not in methods or learning rates.
- **ML researchers / engineers**: people who choose base models and training methods, tune hyperparameters, read loss curves and logs, compare checkpoints and runs, and need every run reproducible from the SDK or CLI.

## Product Purpose

openfinetuner (formerly ShadowLM) is a fine-tuner: load any open model, fine-tune it with any of 13 methods on any hardware, evaluate it, and own the weights. Success for a business user is a fine-tuned model that does their job well enough to ship, proven on their own data; success for a researcher is a fast, inspectable, reproducible fine-tuning loop.

## Positioning

One fine-tuner across the whole matrix: any open model × 13 training methods (from LoRA and QLoRA to DPO, GRPO and MoRE for exact fact recall) × any hardware (CUDA, Apple Silicon, or a fleet of NAT'd workers), with every choice visible and reproducible from the SDK and CLI, and the weights staying with the user.

Shadowing is one concept the fine-tuner supports, not its identity: capturing an agent's real traffic through an OpenAI-compatible proxy (`slm.capture()`) or its OpenTelemetry traces, and fine-tuning on it so a task can move off a rented frontier model without modifying the agent.

## Operating Context

- The studio is the web UI of `shadowlm serve`: one machine trains for real (CUDA/torch in production, mlx on Apple Silicon for development), and NAT'd machines join as workers over one outbound websocket.
- Production runs at studio.shadowlm.sh on a single L40S 48 GB GPU box; demos to enterprise prospects (e.g. a bank's fine-tuning team) happen live in it.
- The studio can be embedded in opencontroller (Lyzr's agent control plane) over the `oc-embed/1` bridge, where opencontroller supplies menu, sign-in, theme and role (viewer / operator / admin).
- Everything in the UI has an SDK and CLI equivalent (`slm.load → finetune → generate → save`, `shadowlm finetune`); the Train page already shows the equivalent CLI command.

## Capabilities and Constraints

- In the UI today: datasets (upload JSONL, Hugging Face import, preview), a model catalog with downloads, a 4-step training wizard (data → model → method → tune), runs with live metrics, logs, checkpoints and adapter download, a Playground with fine-tune-vs-base comparison and checkpoint selection, and machines/worker tokens.
- Agent capture, trace import, frontier comparison and serving are in the studio too: an agent's traffic passes through `/v1/capture/<id>` to the user's frontier model and is recorded into a dataset; OpenTelemetry GenAI traces import as a dataset; evaluations can include the frontier model as a baseline and use it as the judge; and a fine-tune is served at `/openai/v1` (OpenAI-compatible, its own key per deployment). Prompt optimization (`apo.py`) remains SDK-only.
- Training methods: 13, declared as specs (LoRA, QLoRA, DoRA, full, CPT, DPO, GRPO, MoRE, MoRE+, and others); MoRE is the method for exact fact recall.
- One local training job at a time; inference models are cached on the GPU and freed before training.
- Requests through Cloudflare time out after 100 seconds; long operations must not depend on a single held request.
- Terminology: the product's unit is a fine-tune (a trained adapter over a base model). "Shadow" is legacy wording from the shadowing concept; user-facing language should say fine-tune or model, and reserve "shadow" for the agent-capture path and the compare-with-base view.
- Shipping: a fine-tuned model is served at an OpenAI-compatible endpoint (`/openai/v1/chat/completions`) under its deployment's name and key, so an agent switches by changing base URL, key and model name; adapter download stays available. Serving shares the studio's one GPU with training and the Playground, one request at a time.
- Frontier comparison and LLM-judge scoring (decided) use a frontier-model API key the user adds, stored on the server like the Hugging Face token. Without one, evaluation uses rule-based scorers (exact / contains / numeric / JSON) against the dataset's own answers.

## Brand Commitments

- Name: **openfinetuner**, shown with "formerly ShadowLM". The Python package, CLI and imports remain `shadowlm`.
- Visual system: the opencontroller console's design system (Paper & Ink tokens, light and dark, shadcn primitives in `frontend/src/components/ui/`). This redesign keeps it in place.
- The red brain mark stays (`shadowlm/_assets/logo.png`; white `logo1.png` in dark mode).
- Credited as "from Lyzr Research Labs".
- Never name competitor fine-tuning products in code, docs or UI.

## Evidence on Hand

- Real runs on the production box (e.g. `lyzr-assistant`, Qwen3-8B MoRE; earlier `lyzr-support`, `shadow-support`).
- Datasets: `lyzr-faq` (10 facts about Lyzr), `lyzr-support` (50 chat rows), and bundled starter datasets.
- No customer testimonials, benchmarks, cost-savings figures or case studies exist; the UI must not invent them. Any cost or quality comparison must come from the user's own runs.

## Product Principles

1. **Fine-tuning is the product.** Every journey is a fine-tune: data in, a trained model out. Shadowing is one way to bring data, not the frame.
2. **Prove before you ship.** A fine-tune isn't done at "training complete"; it's done when an evaluation on the user's own data shows it beats its base (and the frontier model, when compared).
3. **One studio, two depths.** Business and Research modes work on the same datasets, runs and models; switching modes never hides or forks the truth.
4. **Nothing hidden.** Any configuration the UI chooses is visible and reproducible as SDK or CLI; no silent magic.
5. **You own the weights.** Every successful journey ends with something the user can take away or plug in.
