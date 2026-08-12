# ============================================================
# SKETCHAIR WEB SERVER
# ============================================================

import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PYTHON_DIR = ROOT / "Python"
WEB_DIR = ROOT / "web version"

# Make Python files inside /Python importable
sys.path.insert(0, str(PYTHON_DIR))

# IMPORTANT:
# air_drawing.py only opens the camera / creates a window when
# run directly (python air_drawing.py). Importing it here is
# safe and does not touch any hardware.

import air_drawing  # noqa: E402


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SketchAir",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================
#
# Needed so the frontend can call this backend from a different
# origin, e.g. a Netlify domain. Restrict allow_origins to your
# actual Netlify URL once you have it, instead of "*".
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# SERVE THE FRONTEND (index.html, app.js, style.css)
# ============================================================
#
# Mounted at "/" so relative paths in index.html ("app.js",
# "style.css") resolve the same way whether this FastAPI app
# is serving them, or the "web version" folder is deployed
# as-is to a static host like Netlify.
#
# This must be registered AFTER /health and /ws below, so
# those routes are matched first — otherwise the catch-all
# static mount would shadow them.
# ============================================================


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "online",
        "application": "SketchAir"
    }


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    print("SketchAir browser connected.")

    try:

        while True:

            message = await websocket.receive()

            # ------------------------------------------------
            # CAMERA FRAME
            # ------------------------------------------------

            if message.get("bytes"):

                frame_data = message["bytes"]

                try:

                    # Run the CPU-heavy MediaPipe/OpenCV work in a
                    # worker thread so it never blocks the asyncio
                    # event loop. Without this, one slow frame delays
                    # every other frame and the video appears to
                    # freeze under real-time load.
                    loop = asyncio.get_running_loop()

                    result = await loop.run_in_executor(
                        None,
                        air_drawing.process_frame,
                        frame_data
                    )

                    if result is not None:

                        await websocket.send_bytes(
                            result
                        )

                except Exception as error:

                    print(
                        "Frame processing error:",
                        error
                    )

            # ------------------------------------------------
            # COMMAND
            # ------------------------------------------------

            elif message.get("text"):

                text = message["text"]

                print(
                    "Browser command:",
                    text
                )

                try:

                    payload = json.loads(text)

                    command = payload.get("command")

                    if command:

                        reply = air_drawing.handle_command(
                            command,
                            payload
                        )

                        if reply:

                            await websocket.send_json(
                                reply
                            )

                except json.JSONDecodeError:

                    print(
                        "Ignoring non-JSON command:",
                        text
                    )

                except Exception as error:

                    print(
                        "Command handling error:",
                        error
                    )

    except Exception as error:

        print(
            "WebSocket closed:",
            error
        )


# ============================================================
# STATIC FRONTEND (must come after /health and /ws above)
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=str(WEB_DIR),
        html=True
    ),
    name="static"
)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print("                    SKETCHAIR")
    print("=" * 55)
    print()
    print("Web folder:")
    print(WEB_DIR)
    print()
    print("Open:")
    print("http://127.0.0.1:8000")
    print()
    print("Press CTRL+C to stop.")
    print()
    print("=" * 55)

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )
