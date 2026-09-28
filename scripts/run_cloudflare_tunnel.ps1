# Wusal demo: Cloudflare *Quick Tunnel* on port 8000 only.
# Does NOT start, stop, or edit the Evalia named tunnel (evalia.jehamai.com -> :5001).

$ErrorActionPreference = "Stop"
$cloudflared = @(
    "$env:USERPROFILE\cloudflared\cloudflared.exe",
    "$env:ProgramFiles\cloudflared\cloudflared.exe",
    (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $cloudflared) {
    Write-Host "cloudflared not found. Install: winget install Cloudflare.cloudflared"
    exit 1
}

$port = if ($env:WUSAL_PORT) { $env:WUSAL_PORT } else { "8000" }
$target = "http://127.0.0.1:$port"
$isolatedConfig = Join-Path $PSScriptRoot "cloudflare-wusal-quick.yml"

if ($port -eq "5001") {
    Write-Host "Refusing port 5001 — that is reserved for Evalia (evalia.jehamai.com)."
    exit 1
}

try {
    Invoke-WebRequest -Uri $target -UseBasicParsing -TimeoutSec 3 | Out-Null
} catch {
    Write-Host "Nothing is listening on $target — start Wusal first:"
    Write-Host "  cd C:\Users\Jeham\signlanguage"
    Write-Host "  & 'C:\Users\Jeham\gpu-env\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port $port"
    exit 1
}

Write-Host ""
Write-Host "Wusal Quick Tunnel -> $target"
Write-Host "Evalia tunnel is NOT touched (evalia.jehamai.com stays on localhost:5001)."
Write-Host "Uses isolated config: $isolatedConfig"
Write-Host "Share the https://*.trycloudflare.com URL only for the demo; Ctrl+C to stop THIS tunnel only."
Write-Host ""

& $cloudflared tunnel --config $isolatedConfig --url $target
