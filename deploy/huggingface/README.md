---
title: Synthetic.API
emoji: 🔗
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Synthetic.API — the API-Less Bridge

A multi-agent swarm that acts as a **synthetic API for the open web** —
ask a question, three coordinating agents search, fetch, embed, and draft
a cited answer; describe a task on a real page (a signup form, a login
wall), and a vision-guided browser loop carries it out live, pausing to
ask you for credentials when a page is gated rather than guessing or
storing them.

Built solo for **The Dawn of the Autonomous AI Builder** (Lyzr × Qdrant ×
Omi), Collaborative Multi-Agent Workflows track.

Full source, architecture, and the local docker-compose setup:
[github.com/Oruwe/Synthetic.API](https://github.com/Oruwe/Synthetic.API)

This Space is a single-container adaptation of that same system (see
`deploy/huggingface/SETUP.md` in the repo for exactly what's the same and
what's different here) — built to run on Hugging Face's genuinely free
tier with no credit card required.
