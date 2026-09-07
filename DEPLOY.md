# Deploying Synthetic.API (Oracle Cloud Always Free)

This deploys the exact `docker-compose.yml` stack this repo already runs
locally — nine containers, unchanged in shape — onto a real, persistent,
genuinely-free VM. No architecture rewrite: if `docker compose up --build`
works on your machine, the same command is most of this guide.

**Read this first, honestly:** the free Ampere A1 shape Oracle offers is
**ARM64**, not the x86_64 most of this stack has actually been run on
during development. Every base image here (`python:3.11-slim`,
`qdrant/qdrant`, `redis:7-alpine`, `postgres:16-alpine`) publishes ARM64
builds, and Python's own dependencies (FastAPI, Qdrant client, FastEmbed,
Playwright, etc.) all ship ARM64 wheels — so this *should* build and run
cleanly. But it has not been verified on real ARM hardware before this
guide was written. **Step 6 below is the actual verification, not a
formality** — do not assume success until every service's logs confirm it.

## 1. Create the VM

1. Sign up / log in at [cloud.oracle.com](https://cloud.oracle.com). Free
   tier signup asks for a card — Always Free resources are never charged,
   but the signup flow itself requires one on file.
2. **Compute → Instances → Create Instance.**
3. **Image and shape:** change the shape to **Ampere (ARM)** →
   `VM.Standard.A1.Flex`, and set it to the Always Free maximum: **4
   OCPUs, 24 GB memory**. Image: **Ubuntu 22.04** (the Canonical Ubuntu
   ARM image, not the "Oracle Linux" default).
4. **Networking:** use a new or existing VCN, keep "assign a public IPv4
   address" checked.
5. **SSH keys:** upload your own public key (or generate one in-console
   and download the private key — you cannot retrieve it again later).
6. **Boot volume:** the default (~50GB) is fine; Always Free covers up to
   200GB total across your boot volumes.
7. Create the instance and note its **public IP address**.

If Ampere A1 capacity shows as unavailable in your chosen region (a real,
commonly-reported Oracle Free Tier constraint, not a mistake on your
part), try a different availability domain in the same region, or a
neighboring region — capacity availability fluctuates.

## 2. Open the right ports

Two separate firewalls both need opening — a common gotcha, since fixing
only one silently doesn't work.

**a) Oracle's cloud-level firewall (Security List / Network Security
Group):** in the console, open your VCN's default Security List (or the
NSG attached to this instance) → **Add Ingress Rules**. Add TCP rules for:

| Port | Purpose |
|---|---|
| 22 | SSH (likely already open by default) |
| 8000 | Orchestrator API |
| 7860 | Demo UI |

Deliberately **not** opening 5000, 5050, 6333, 6334, 3000, 5432, 6379 —
`docker-compose.yml` binds all of those to the VM's own loopback only
(see its comments), so they're not reachable from outside the VM
regardless of this firewall; there's nothing to open for them.

**b) The VM's own host firewall (iptables/ufw):** Oracle's stock Ubuntu
images ship with a restrictive default iptables ruleset independent of
the cloud console. SSH in and run:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 7860 -j ACCEPT
sudo netfilter-persistent save   # persist across reboots; install it first if missing:
# sudo apt-get install -y iptables-persistent
```

(If this VM instead uses `ufw`, use `sudo ufw allow 8000/tcp` and
`sudo ufw allow 7860/tcp` instead — check `sudo ufw status` /
`sudo iptables -L` to see which applies to your image.)

## 3. Install Docker

```bash
ssh ubuntu@<your-vm-public-ip>

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker   # or log out and back in
docker compose version   # confirms the compose plugin came with it
```

## 4. Get the code and configure secrets

```bash
git clone https://github.com/Oruwe/Synthetic.API.git
cd Synthetic.API
git checkout feature/ambient-rpa-action-bridge   # or main, once merged

cp .env.example .env
nano .env   # or vim/whatever's available
```

Fill in, at minimum:
- `TAVILY_API_KEY`, `OPENROUTER_API_KEY` — the research/action paths do
  nothing useful without these.
- **`ORCHESTRATOR_API_KEY`** — generate one and set it:
  ```bash
  python3 -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
  This is the single most important line in this whole guide. Without it,
  `/trigger` and `/runs/*` are open to anyone who finds your VM's IP —
  they could trigger arbitrary browser actions (the ambient RPA action
  path included) or read/answer any run's paused state. `ui/app.py` picks
  this up automatically via `docker-compose.yml`'s passthrough — set it
  here once, nothing else to configure.
- `POSTGRES_PASSWORD`, `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT` —
  generate each the same way as above. Langfuse is loopback-only per
  `docker-compose.yml` either way, but don't leave a well-known default
  from a public repo as the only thing standing behind it.
- `LYZR_API_KEY`/`LYZR_AGENT_ID` (optional — see README's "Running it"
  for the Lyzr Studio setup; the system falls back to OpenRouter without
  it), `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (generate these from
  Langfuse's own UI after it's running — see step 9).

**Never commit `.env`.** It's already gitignored; double check before
pushing anything from this VM.

## 5. Bring it up

```bash
docker compose up --build -d
```

The first build compiles Playwright's Chromium download and the
FastEmbed model fetch — expect several minutes, longer than you're used
to locally if this is the VM's first build.

## 6. Verify — do not skip this

```bash
docker compose ps
```

Every service should show `Up` (most with `(healthy)`). If anything is
`Restarting` or exits immediately, that's the ARM64 risk flagged at the
top of this guide manifesting — check its logs first:

```bash
docker compose logs <service-name>
```

Then confirm the actual product works, from your own machine (not the VM):

```bash
curl -s http://<vm-public-ip>:8000/health
curl -s http://<vm-public-ip>:8000/ready | python3 -m json.tool
curl -s -X POST http://<vm-public-ip>:8000/trigger \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your ORCHESTRATOR_API_KEY>" \
  -d '{"transcript": "what is the latest on the ISRO hackathon rules"}'
```

That should return a `run_id`. Poll `GET /runs/<run_id>` (same
`X-API-Key` header) until `answer_text` is populated, or just open
`http://<vm-public-ip>:7860` in a browser and ask it there.

## 7. Point a real Omi at this deployment

Everything above proves the product works over `curl`/the UI — but the
actual point of this build is that Omi's own wearable/app is the input,
not a text box (see README's "Voice & UI" section). Wiring a real Omi
device or app at this VM is what makes this a real "synthetic API for
Omi" deployment rather than a demo of one:

1. **Set a real webhook secret**, if you haven't already — `OMI_WEBHOOK_SECRET`
   in `.env`, same generation pattern as the others (`python3 -c
   "import secrets; print(secrets.token_urlsafe(32))"`). Unlike
   `ORCHESTRATOR_API_KEY`, this gates only `/webhook/omi` specifically
   (`agents/orchestrator/omi_webhook.py::verify_webhook_secret`) — it's a
   separate credential because it's presented by Omi's own infrastructure,
   not by you.
2. In whatever Omi lets you configure as a webhook target (the Omi app's
   integration/developer settings, or the hackathon starter kit's own
   config, per `agents/orchestrator/omi_webhook.py`'s own
   TODO(verify) note — the exact payload shape and auth header Omi sends
   are not independently confirmed here from first-hand docs, only
   inferred from publicly documented examples, so cross-check against
   the actual starter kit before relying on this in a live demo), point
   the webhook URL at:
   ```
   http://<vm-public-ip>:8000/webhook/omi
   ```
3. Restart the orchestrator to pick up the new secret if you set one
   after first bringing the stack up:
   ```bash
   docker compose up -d agents-orchestrator
   ```
4. Speak a task-shaped or question-shaped utterance to the Omi device and
   watch it arrive:
   ```bash
   docker compose logs -f agents-orchestrator
   ```
   A `run_id` should appear in the logs the moment Omi's POST lands,
   exactly like the `curl` trigger in step 6 above — from Omi's webhook
   onward, it's the identical DAG/executor/Synthesizer path, so anything
   already verified there (research answers, the ambient RPA action path,
   the human-in-the-loop gated-content pause/resume flow) works the same
   whether the transcript came from `curl`, the UI, or a real Omi device.
5. If nothing arrives: confirm Omi's dashboard shows the POST as
   succeeding (not silently retried/dropped against a timeout — Omi's own
   delivery UI, if it has one, is the first place to look, before
   assuming this stack is at fault), and check
   `docker compose logs agents-orchestrator` for a 401 (wrong/missing
   secret — case 1 above) or a `could not extract a transcript` error
   (the payload shape didn't match `parse_omi_payload`'s assumptions —
   this is exactly the TODO(verify) risk flagged above; paste the real
   payload shape from Omi's logs and `parse_omi_payload()` is a small,
   isolated function to adjust, not a redesign).

## 8. Survive a reboot

`restart: unless-stopped` on every service (already in `docker-compose.yml`)
means Docker restarts them if they crash or the VM reboots — but only if
the Docker *daemon itself* starts on boot:

```bash
sudo systemctl enable docker
```

(`get.docker.com`'s installer usually enables this already — confirm with
`systemctl is-enabled docker` rather than assuming.)

## 9. Optional: Langfuse setup, and a domain + HTTPS

**Langfuse:** it's bound to loopback only, so reach it via an SSH tunnel:

```bash
ssh -L 3000:localhost:3000 ubuntu@<your-vm-public-ip>
```

Then open `http://localhost:3000` in your own browser, sign up (this
first account becomes the instance's admin — nothing here is shared with
any real Langfuse service), create a project, and generate an API
keypair under its settings. Put those into `.env` as
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, then
`docker compose up -d` again to pick them up (tracing degrades gracefully
without them either way — see README's observability section).

**HTTPS:** this guide deliberately ends on plain HTTP over the VM's public
IP — acceptable for a hackathon demo link, not something to point real
users' credentials at long-term (the human-in-the-loop gated-content
feature's email/password prompt included). If you have a domain, point an
A record at the VM's public IP and put a reverse proxy (Caddy is the
simplest — one line of config gets you automatic Let's Encrypt HTTPS) in
front of ports 8000 and 7860; that's a deliberate follow-up, not something
this guide builds blind without knowing whether you have a domain to use.

## What this does *not* change

Everything else about this system — the DAG executor, the credential
handling in the human-in-the-loop path, the ambient RPA action loop's
safety rails, the test suite — is identical to what runs locally. This
guide is entirely about getting the same, unmodified stack onto a real
VM safely; see the main [README](README.md) for how the system itself
works.
