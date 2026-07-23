"""Modal deployment of the linprogx demo API.

Serves the same network-flow FastAPI app (`demo.api.main:app`) that Render
ran, but on Modal: container starts in seconds (vs the 30-60s spin-up of a
Render free instance) and scales to zero so idle cost is ~nothing. The public
site's Cloudflare Pages proxy (`/api/lp`) points its origin here; Render stays
available as a config-flip rollback (LINPROGX_ORIGIN env var on the proxy).

Deploy:  uv run --with modal --no-project modal deploy deploy/modal_app.py

The image ships the pure-Python solver source — the guarded `_cfast`/`_csparse`
imports fall back to `_fast.py` cleanly, and demo-scale problems (<= 20 nodes,
<= 50 edges) solve in well under a millisecond either way, so skipping the C
build keeps the image tiny and cold starts fast.

No DEMO_SHARED_SECRET is set here, so the app's optional X-Demo-Secret check
is a no-op (mirrors the TourneyDesk demo's keyless /solve). The API's own
protections still apply: 30 req/min/IP rate limit, size caps, 5 s solve cap.

Cost knob: `min_containers=0` = scale-to-zero. Set to 1 for always-warm if
first-hit latency ever matters more than idle cost; `scaledown_window` keeps
a warmed container around for up to Modal's 20-minute maximum so clustered
visits don't each cold-start.
"""

from __future__ import annotations

import modal  # ty: ignore[unresolved-import] -- optional deploy-only dependency

# Pins match demo API requirements (see render-build.sh / README-DEMO.md).
# The solver itself is dependency-free; its source is copied in directly.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("fastapi==0.115.12", "pydantic==2.11.3")
    .add_local_dir("src/linprogx", remote_path="/root/linprogx")
    .add_local_python_source("demo")
)

app = modal.App("linprogx-demo")


@app.function(
    image=image,
    min_containers=0,  # scale-to-zero; set to 1 for always-warm
    # Modal's maximum idle window: absorb clustered personal-site visits while
    # retaining scale-to-zero instead of paying for permanent warm capacity.
    scaledown_window=1200,
    # Restore from a memory snapshot instead of cold-importing Python on
    # scale-from-zero — measured ~20 s worst-case cold starts drop to a few
    # seconds. The app is snapshot-safe: no GPU, no open sockets at import.
    enable_memory_snapshot=True,
    cpu=1.0,
    memory=512,  # dense simplex at demo scale is tiny
    timeout=30,  # app enforces its own 5 s solve cap well inside this
)
# Solves are sub-millisecond, so one container can serve many requests
# concurrently; Modal adds a container once ~8 are in flight.
@modal.concurrent(max_inputs=12, target_inputs=8)
@modal.asgi_app()
def fastapi_app():
    from demo.api.main import app as demo_app

    return demo_app
