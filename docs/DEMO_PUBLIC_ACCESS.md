# Public demo access (Cloudflare & alternatives)

The PoC runs on your PC (GPU + local models). For **SDAIA or remote viewers**, expose **`http://127.0.0.1:8000`** with a tunnel—no router port forwarding.

## Do not disturb Evalia

Your **Evalia** app uses a **named tunnel** in `%USERPROFILE%\.cloudflared\config.yml`:

- Tunnel: `jehamai-api`
- `evalia.jehamai.com` → `http://localhost:5001`

**For Wusal, only use** `scripts/run_cloudflare_tunnel.ps1`. It:

- Starts a separate **Quick Tunnel** (random `trycloudflare.com` URL) to **port 8000 only**
- Uses **`scripts/cloudflare-wusal-quick.yml`** — **not** your Evalia `config.yml`
- Does **not** run `cloudflared tunnel run` or edit Evalia ingress

**Do not** add Wusal to `~/.cloudflared/config.yml` unless you intentionally want a second hostname on the same tunnel (e.g. `wusal.jehamai.com` → `:8000`) and know how to reload the Evalia tunnel service safely.

| App | Port | Public URL |
|-----|------|------------|
| Evalia | 5001 | `https://evalia.jehamai.com` (existing) |
| Wusal | 8000 | Quick Tunnel URL (temporary, per session) |

## Recommended: Cloudflare Quick Tunnel (free)

You already have `cloudflared` at `C:\Users\Jeham\cloudflared\cloudflared.exe`.

**Terminal 1 — app**

```powershell
cd C:\Users\Jeham\signlanguage
& 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — tunnel**

```powershell
cd C:\Users\Jeham\signlanguage
.\scripts\run_cloudflare_tunnel.ps1
```

Cloudflare prints a URL like `https://something-random.trycloudflare.com` — share **only with meeting attendees**. Stopping this script **only** stops the Wusal quick tunnel; Evalia keeps running.

If you see **HTTP 404** on the trycloudflare link, the tunnel config was routing to `http_status:404`. Use the project file `scripts/cloudflare-wusal-quick.yml` (forwards to port **8000**), then **restart** the tunnel script so Cloudflare gives a **new** URL—the old link will not fix itself.

| Pros | Cons |
|------|------|
| Free HTTPS in minutes | URL **changes** each run (Quick Tunnel) |
| No open firewall ports | Your PC must stay on for the demo |
| Works with local GPU/models | Uses your **OpenAI key** and **upload bandwidth** |

### Stable URL (optional, for repeated demos)

1. Create a free [Cloudflare](https://dash.cloudflare.com) account.
2. Add a domain (or use a subdomain you control).
3. Create a **named tunnel** in Zero Trust → Networks → Tunnels.
4. Point `wusal.yourdomain.com` → `http://localhost:8000`.

Docs: [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)

Add **Cloudflare Access** (email OTP) in front of the tunnel if the link might leak.

---

## Alternatives

| Service | Best for |
|---------|----------|
| **[ngrok](https://ngrok.com)** | Same as Quick Tunnel; free tier, random URL |
| **Tailscale Funnel** | Private mesh + optional public HTTPS |
| **Railway / Render / Fly.io** | 24/7 hosted app — **hard** here (GPU, large KArSL files, local `.env`) |
| **Azure / GCP VM with GPU** | Production pilot; not needed for a 1-hour IP demo |

---

## Security (read before sharing a link)

- Anyone with the URL can call **`/api/sign-conversation`** → **LLM charges** on your key.
- Uploaded videos pass through your machine; treat as **demo-only**, not production.
- Do **not** commit `.env` or share tunnel URLs on social media.
- After the meeting: **stop the tunnel** (Ctrl+C) and stop uvicorn.

---

## Quick test

With tunnel running, open the `trycloudflare.com` URL on your phone (Wi‑Fi). Upload a short test clip to confirm video upload size and latency.
