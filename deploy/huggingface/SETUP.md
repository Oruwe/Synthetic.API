# Deploying Synthetic.API to Hugging Face Spaces (free, no credit card)

> **Currently blocked, flagged honestly:** as of ~July 2026, Hugging Face
> marked the Docker SDK as a **paid** feature when creating a new Space
> (confirmed on HF's own community forums; described there as possibly
> temporary, tied to an infrastructure shortage). This guide is kept
> ready and working — check whether Docker SDK is free again before
> following it, and if it's still gated, see
> **[deploy/render/SETUP.md](../render/SETUP.md)** instead, which runs
> this exact same single-container build (`deploy/common/`) on Render's
> free tier, no card, confirmed working as of this writing.

This is the deployment path for when a real cloud VM isn't an option —
Oracle/AWS/GCP/Azure all require a credit card even on their free tiers,
and Azure for Students' verification doesn't work for every institution.
Hugging Face Spaces' free CPU tier needs neither: no card, and it doesn't
sleep until 48 hours of inactivity (not 15 minutes, like some other
card-free options).

**Read this first, honestly — what's different here vs. the main
`DEPLOY.md` (Oracle VM) path:**

- HF Spaces runs **one container**, not docker-compose's nine services.
  Everything here is bundled into a single image, supervised by
  `start.sh`: the merged Orchestrator+UI app (see `merged_app.py`),
  `demo_target` (the ambient-RPA/human-in-the-loop fixture pages), and
  `agents-synthesizer`.
- **Qdrant is not self-hosted here.** This points at a free Qdrant Cloud
  cluster instead of running the qdrant server in-container — see step 1.
- **Langfuse, Postgres, and Redis are not run at all** in this
  deployment. Both are already designed to fail open (the pipeline works
  identically without them — see `agents/common/langfuse_tracer.py`'s and
  `REDIS_URL`'s own comments in `.env.example`), so this isn't a
  compromise on the actual system, just a simplification for a
  single-container host. You still get the trace UI locally via
  docker-compose if you want it.
- **Storage is ephemeral.** HF Spaces' free tier has no persistent disk.
  This is fine for what a judge actually does: every question triggers a
  fresh, live search → fetch → embed → answer cycle regardless of what's
  already stored, so nothing about the demo depends on old data
  surviving a restart or the 48-hour sleep/wake cycle.

## 1. Create a free Qdrant Cloud cluster

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) — no credit card
   needed for the free tier (0.5 vCPU / 1GB RAM / 4GB disk, permanent,
   fits roughly a million 768-dim vectors — comfortably enough for this).
2. Create a cluster (any region). Once it's up, copy its **cluster URL**
   (looks like `https://xxxxxxxx.region.cloud.qdrant.io:6333`) and
   generate an **API key** under the cluster's access management.
3. One real limitation, worth knowing: a free cluster **suspends after 1
   week of inactivity** and is **deleted after 4 weeks** if never
   reactivated. Fine for an active hackathon judging window; if this sits
   idle for a while afterward, just log back into the dashboard and
   reactivate it (a few clicks) before your next demo.

## 2. Push this deployment to a new Hugging Face Space

1. Sign up at [huggingface.co](https://huggingface.co) — no credit card
   for CPU Basic hardware.
2. **New Space** → give it a name → SDK: **Docker** → hardware: **CPU
   basic (free)**.
3. HF gives you a new, empty git repo for the Space. Clone it locally,
   then copy in exactly two files from this repository:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   cp /path/to/Synthetic.API/deploy/huggingface/Dockerfile .
   cp /path/to/Synthetic.API/deploy/huggingface/README.md .
   git add Dockerfile README.md
   git commit -m "Deploy Synthetic.API"
   git push
   ```
   That's genuinely all the Space's own repo needs — its `Dockerfile`
   `git clone`s the real source from GitHub at build time (see the
   `SOURCE_REPO`/`SOURCE_REF` build args in that file), so this Space
   never has to mirror the full codebase.
4. The Space starts building automatically on push. **First build is
   slow** (Playwright's Chromium download + the FastEmbed model fetch,
   same as the Oracle path) — watch the build logs in the Space's own
   "Logs" tab rather than assuming it's stuck.

## 3. Set your secrets

In the Space's **Settings → Repository secrets**, add:

- `TAVILY_API_KEY`, `OPENROUTER_API_KEY` — required; the research/action
  paths do nothing useful without these.
- `QDRANT_URL`, `QDRANT_API_KEY` — from step 1.
- `ORCHESTRATOR_API_KEY` — **strongly recommended**. This Space's public
  URL is reachable by anyone; without this, anyone who finds it can hit
  `/trigger` directly and burn through your Tavily/OpenRouter quota, or
  drive the ambient RPA action path. Generate one with
  `openssl rand -base64 32 | tr -d '\n='`. The UI still works perfectly
  for a judge clicking around normally — it already knows this same
  secret internally (see `merged_app.py`) — this only blocks someone
  bypassing the UI and hitting the raw API directly.
- `OMI_WEBHOOK_SECRET` — optional, only if you're wiring a real Omi
  device at this deployment (see step 5 below).
- `LYZR_API_KEY` / `LYZR_AGENT_ID` — optional (falls back to OpenRouter
  without it, same as everywhere else in this project).

Setting a secret triggers an automatic rebuild.

## 4. Verify — don't assume it worked

Once the build finishes and the Space shows "Running", open its URL
directly — you should see the same custom dark UI this project runs
locally, not Hugging Face's default Space chrome (Docker SDK Spaces
serve your app directly).

Ask it something in the UI (e.g. *"What's the latest on the ISRO
hackathon rules?"*) and confirm you get a real, cited answer back.

To also exercise the ambient RPA / human-in-the-loop pause-resume flow
live (the flagship feature of this build), ask something that points at
the bundled fixture pages, reachable from *inside* this container at
`localhost:5050` (not separately exposed — this is what `demo_target` is
for, see the main README's "Human-in-the-loop" section):

> "Go to http://localhost:5050/members and log in with
> demo@example.com / demo123, then tell me what the members page says."

That should pause the run, prompt for the credentials in the UI, and
resume once you answer — exactly like the local docker-compose demo,
proven live earlier this session.

If something's wrong, the Space's **Logs** tab shows this container's
real stdout — the same structured JSON logs as everywhere else in this
project, correlated by `run_id`.

## 5. Point a real Omi at this deployment

Same idea as the main `DEPLOY.md`'s equivalent step, adjusted for one
port: Omi's webhook target is

```
https://<your-space-url>/webhook/omi
```

with `OMI_WEBHOOK_SECRET` (step 3) as its shared secret. Everything past
that point — DAG execution, the Synthesizer, the pause/resume flow — is
identical to the `curl`/UI path, since it's the same merged app either
way.

## 6. Updating this deployment later

Since the Space's `Dockerfile` clones GitHub's `main` branch at *build*
time, pushing new code to GitHub does **not** automatically rebuild this
Space. Trigger a rebuild manually: the Space's **Settings → Factory
reboot**, or simply push any trivial commit (even just re-committing the
same `Dockerfile`) to the Space's own tiny repo.

## What this does *not* change

The actual system — the DAG executor, credential handling in the
human-in-the-loop path, the ambient RPA action loop's safety rails, the
full test suite — is identical to what runs locally and on the Oracle VM
path. This file is entirely about fitting that same system into a
single, free, card-free container; see the main
[README](https://github.com/Oruwe/Synthetic.API#readme) for how the
system itself works.
