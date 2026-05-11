"""CLI entrypoint: starts the FastAPI server and opens the UI in the browser."""

import argparse
import threading
import time
import webbrowser

import uvicorn

from livepose.server import app


def main() -> None:
    parser = argparse.ArgumentParser(prog="livepose")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Thread(
            target=lambda: (time.sleep(1.0), webbrowser.open(url)),
            daemon=True,
        ).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
