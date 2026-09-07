# Deploying Synthetic.API to Render (free, no credit card)

This runs the exact same single-container build as the Hugging Face path
(`deploy/common/merged_app.py` + `start.sh`, unmodified) on Render's free
Docker Web Service instead — use this if Hugging Face's Docker SDK is
still gated behind payment when you read this (see
[deploy/huggingface/SETUP.md](../huggingface/SETUP.md)'s note at the top).

**Read this first, honestly:**

- **Render's free tier is 512MB RAM** — much tighter than Hugging Face
  Spaces' 16GB. This has not been confirmed to actually fit (at build
  time or run time) from either sandbox available while writing this.
  Step 5 below (verify) is the real test, not a formality — watch the
  build logs and, once running, actually exercise the ambient-RPA demo
  (the most memory-hungry path, since it launches real Chromium) before
  assuming this works.
- **Free services sleep after 15 minutes of inactivity** and take 30-60
  seconds to cold-start on the next request — tighter than Hugging
  Face's 48-hour window. Fine for a demo you're actively showing;
  worth waking it up a minute before a scheduled judging call if it's
  been idle.
- Same simplifications as the Hugging Face path otherwise: Qdrant is a
  free managed Qdrant Cloud cluster, not self-hosted; Langfuse/Postgres/
  Redis don't run at all (both already fail open by design); storage is
  ephemeral (fine — every question does a fresh live search → fetch →
  embed → answer cycle regardless of old data).

## 1. Create a free Qdrant Cloud cluster

Same as the Hugging Face path — sign up at
[cloud.qdrant.io](https://cloud.qdrant.io) (no card), create a free
cluster (0.5 vCPU / 1GB RAM / 4GB disk, permanent), copy its **cluster
URL** and generate an **API key**. (Suspends after 1 week idle, deleted
after 4 weeks idle — reactivate from the dashboard if that happens.)

## 2. Create the Render Web Service

1. Sign up at [render.com](https://render.com) — no credit card for the
   free instance type.
2. **New → Web Service**, connect this GitHub repository
   (`Oruwe/Synthetic.API`), branch `main`.
3. When asked for the runtime/environment, choose **Docker**. Set:
   - **Dockerfile Path:** `deploy/render/Dockerfile`
   - **Docker Build Context Directory:** `.` (repo root — the Dockerfile
     needs the whole tree, not just `deploy/render/`)
4. **Instance Type:** Free.
5. Create the service. The first build is slow (Playwright's Chromium
   download + the FastEmbed model fetch) — watch Render's build logs
   rather than assuming it's stuck. **This build is also the first real
   test of the 512MB risk flagged above** — if it fails partway through
   `uv sync` or the Chromium install with an out-of-memory-shaped error,
   that's this risk manifesting; nothing else to debug first.

## 3. Set your environment variables

In the service's **Environment** tab, add:

- `TAVILY_API_KEY`, `OPENROUTER_API_KEY` — required.
- `QDRANT_URL`, `QDRANT_API_KEY` — from step 1.
- `ORCHESTRATOR_API_KEY` — **strongly recommended**, same reasoning as
  the Hugging Face path: this service's URL is public, and without this
  anyone can hit `/trigger` directly and burn your API quota. Generate
  with `openssl rand -base64 32 | tr -d '\n='`. The UI still works fine
  for a normal visitor — it already knows this same secret internally.
- `OMI_WEBHOOK_SECRET` — optional, only if wiring a real Omi device
  (step 5).
- `LYZR_API_KEY` / `LYZR_AGENT_ID` — optional.

Saving triggers a redeploy (reusing the already-built image layers where
possible, so this is faster than the first build).

## 4. Updating this deployment later

Unlike the Hugging Face path (which needs a manual rebuild trigger since
its Dockerfile clones GitHub at build time), Render builds directly from
this connected repo — by default it **auto-deploys on every push** to
the connected branch. Turn this off in the service's Settings if you'd
rather deploy manually.

## 5. Verify — don't assume it worked

Once the service shows "Live", open its `https://<service>.onrender.com`
URL — you should see this project's own custom dark UI directly (Render
serves your container's port 7860 at the service's root).

Ask it a real question and confirm you get a cited answer back. Then
exercise the ambient RPA / human-in-the-loop demo — the actual proof
Chromium works within 512MB — by asking something that points at the
bundled fixture pages, reachable from *inside* this container:

> "Go to http://localhost:5050/members and log in with
> demo@example.com / demo123, then tell me what the members page says."

That should pause the run, prompt for credentials in the UI, and resume
once answered. If it hangs or the service restarts mid-run, check the
**Logs** tab for an out-of-memory kill — that's the 512MB risk, and the
fix at that point is either accepting the research-only path works but
the browser-automation path doesn't on this tier, or moving to whichever
platform (Hugging Face, once/if its Docker SDK is free again; or a real
VM per the main `DEPLOY.md`) has more headroom.

## 6. Point a real Omi at this deployment

Same as the Hugging Face path, adjusted for Render's URL:

```
https://<your-service>.onrender.com/webhook/omi
```

with `OMI_WEBHOOK_SECRET` (step 3) as the shared secret.

## What this does *not* change

The actual system is identical to every other deployment path in this
repo — see the main [README](https://github.com/Oruwe/Synthetic.API#readme)
for how it works, and [deploy/huggingface/SETUP.md](../huggingface/SETUP.md)
for the parts of this single-container approach that are genuinely
shared (they're literally the same files, `deploy/common/`) rather than
duplicated per platform.
