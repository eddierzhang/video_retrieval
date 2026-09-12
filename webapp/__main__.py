"""Run the Moments web app: python -m webapp [--port 8765] [--no-browser]"""
import argparse
import threading
import webbrowser

import uvicorn

from .server import LOCAL_HOSTS, create_app


def main():
    parser = argparse.ArgumentParser(description="Search your videos in natural language, fully on this machine.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"Moments is running at {url}")
    # Loopback only: the app has no authentication.
    uvicorn.run(create_app(allowed_hosts=LOCAL_HOSTS), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
