#!/usr/bin/env python3
"""Start the F1 AI Copilot API and its web UI together (Windows, macOS and Linux).

    python scripts/run_app.py                        # API on :8000, UI on :8501, opens the browser
    python scripts/run_app.py --ui-port 8600 --no-browser
    python scripts/run_app.py --host 0.0.0.0         # reachable from other devices (no authentication!)

Both servers run as child processes of this interpreter (``python -m uvicorn`` and
``python -m streamlit``), so the active virtual environment is used. The UI is told
where the API is through F1_API_URL. Ctrl+C (or SIGTERM) stops both; if either one
exits on its own, the other is stopped too and the exit code is non-zero. On Linux the
servers also stop when the launcher itself is killed (e.g. ``kill -9``).
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"
ALL_INTERFACES = ("0.0.0.0", "::", "")
# bind() errors meaning "another program has this port" (Windows: WSAEADDRINUSE, or WSAEACCES
# for a port held exclusively or reserved by the system).
ADDRESS_IN_USE = {errno.EADDRINUSE, *(getattr(errno, name) for name in ("WSAEADDRINUSE", "WSAEACCES") if hasattr(errno, name))}
PR_SET_PDEATHSIG = 1  # linux/prctl.h
# Local health checks must never go through an HTTP proxy from the environment.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class StopRequested(Exception):
    """SIGTERM (or SIGHUP/SIGBREAK) received."""


class ChildExited(Exception):
    def __init__(self, name: str, returncode: int) -> None:
        super().__init__(f"The {name} exited unexpectedly (exit code {returncode})")
        self.returncode = returncode


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a port number") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{port} is outside 1-65535")
    return port


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the F1 AI Copilot API and web UI together.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface both servers listen on (default 127.0.0.1: this computer only). "
        "0.0.0.0 makes them reachable from your network; the API has no authentication.",
    )
    parser.add_argument("--api-port", type=port_number, default=8000, help="API port (default 8000)")
    parser.add_argument("--ui-port", type=port_number, default=8501, help="Web UI port (default 8501)")
    parser.add_argument("--no-browser", dest="open_browser", action="store_false", help="Do not open a browser tab")
    parser.add_argument(
        "--startup-timeout", type=float, default=120.0, help="Seconds to wait for both servers to answer (default 120)"
    )
    parser.add_argument(
        "--stop-timeout", type=float, default=10.0, help="Seconds to wait for a graceful stop before killing (default 10)"
    )
    args = parser.parse_args(argv)
    args.host = args.host.strip().strip("[]")  # [::1] -> ::1
    if args.api_port == args.ui_port:
        parser.error("--api-port and --ui-port must differ")
    if args.startup_timeout <= 0 or args.stop_timeout <= 0:
        parser.error("timeouts must be positive")
    return args


def local_url(host: str, port: int) -> str:
    """URL for reaching a server bound to ``host`` from this computer."""

    host = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def lan_address() -> Optional[str]:
    """This computer's IPv4 address on its network, if it has one (a UDP connect sends no packet)."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("192.0.2.1", 9))  # TEST-NET-1: only selects the outgoing interface
            address = sock.getsockname()[0]
        except OSError:
            return None
    return None if address.startswith("127.") or address == "0.0.0.0" else address


def port_in_use(host: str, port: int) -> bool:
    """Whether another program has ``port`` on ``host``; OSError when ``host`` is not an address of this computer."""

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        if not IS_WINDOWS:  # like the servers themselves: ignore connections in TIME_WAIT
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            if exc.errno in ADDRESS_IN_USE:
                return True
            raise
    return False


def identify_server(url: str) -> Optional[str]:
    """Name the F1 AI Copilot server answering at ``url`` (an earlier run still alive), if it is one."""

    try:
        with _OPENER.open(f"{url}/", timeout=0.5) as response:
            if json.load(response).get("message") == "F1 AI Copilot API":
                return "an F1 AI Copilot API"
    except (OSError, ValueError, AttributeError):
        pass
    try:
        with _OPENER.open(f"{url}/_stcore/health", timeout=0.5) as response:
            if response.read().strip() == b"ok":
                return "a Streamlit app"
    except OSError:
        pass
    return None


def busy_port_message(host: str, port: int, label: str, option: str) -> str:
    owner = identify_server(local_url(host, port))
    finder = f"`netstat -ano | findstr :{port}`" if IS_WINDOWS else f"`lsof -i :{port}`"
    culprit = f" by {owner}; an earlier run of this launcher may still be running" if owner else ""
    return (
        f"Port {port} on {host} is already in use{culprit}. Stop that program (find it with {finder}) "
        f"or pick another {label} port with {option}."
    )


def api_command(host: str, port: int) -> List[str]:
    return [sys.executable, "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)]


def ui_command(host: str, port: int) -> List[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(Path("ui") / "streamlit_app.py"),
        "--server.address", host, "--server.port", str(port),
        "--server.headless", "true", "--browser.gatherUsageStats", "false",
    ]  # fmt: skip


def _stop_with_launcher() -> Optional[Callable[[], None]]:
    """Linux: a ``preexec_fn`` that makes the child receive SIGTERM when the launcher dies, even from SIGKILL."""

    if not sys.platform.startswith("linux"):
        return None
    prctl = ctypes.CDLL(None, use_errno=True).prctl  # resolved here, not in the forked child
    launcher = os.getpid()

    def request_signal() -> None:
        prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        if os.getppid() != launcher:  # the launcher died before prctl took effect
            os._exit(1)

    return request_signal


def start(command: List[str], env: Dict[str, str]) -> subprocess.Popen:
    # Own process group/session: Ctrl+C reaches only this launcher, which stops each child exactly once.
    isolation: Dict[str, Any]
    if IS_WINDOWS:
        isolation = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        isolation = {"start_new_session": True, "preexec_fn": _stop_with_launcher()}
    return subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, **isolation)


def check_alive(children: Dict[str, subprocess.Popen]) -> None:
    for name, process in children.items():
        if process.poll() is not None:
            raise ChildExited(name, process.returncode)


def wait_until_ready(url: str, children: Dict[str, subprocess.Popen], deadline: float) -> None:
    while True:
        check_alive(children)
        try:
            with _OPENER.open(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"{url} did not answer in time")
        time.sleep(0.25)


def stop(children: Dict[str, subprocess.Popen], timeout: float) -> None:
    """Ask every running child to stop, then kill whatever is still running after ``timeout``."""

    for process in children.values():
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGTERM)
            except OSError:  # exited in the meantime
                pass
    deadline = time.monotonic() + timeout
    for name, process in children.items():
        try:
            process.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f"The {name} did not stop within {timeout:g} s; killing it.", file=sys.stderr, flush=True)
            if IS_WINDOWS:
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _raise_stop(signum: int, frame: object) -> None:
    raise StopRequested()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for label, option, port in (("API", "--api-port", args.api_port), ("UI", "--ui-port", args.ui_port)):
        try:
            busy = port_in_use(args.host, port)
        except OSError as exc:
            print(
                f"Cannot listen on --host {args.host}: {exc}. Use an address of this computer "
                "(default 127.0.0.1; 0.0.0.0 for all interfaces).",
                file=sys.stderr,
            )
            return 2
        if busy:
            print(busy_port_message(args.host, port, label, option), file=sys.stderr)
            return 1
    api_url, ui_url = local_url(args.host, args.api_port), local_url(args.host, args.ui_port)
    env = {**os.environ, "F1_API_URL": api_url}
    handled = [getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK") if hasattr(signal, name)]
    previous = {signum: signal.getsignal(signum) for signum in handled}
    for signum in handled[1:]:  # SIGINT keeps raising KeyboardInterrupt
        signal.signal(signum, _raise_stop)

    children: Dict[str, subprocess.Popen] = {}
    exit_code = 0
    try:
        children["API"] = start(api_command(args.host, args.api_port), env)
        children["web UI"] = start(ui_command(args.host, args.ui_port), env)
        deadline = time.monotonic() + args.startup_timeout
        wait_until_ready(f"{api_url}/", children, deadline)
        wait_until_ready(f"{ui_url}/_stcore/health", children, deadline)
        print(
            f"\nF1 AI Copilot is running\n"
            f"  Web UI: {ui_url}  (pid {children['web UI'].pid})\n"
            f"  API:    {api_url}  (docs: {api_url}/docs, pid {children['API'].pid})\n"
            "Press Ctrl+C to stop both.\n",
            flush=True,
        )
        if args.host in ALL_INTERFACES:
            lan = lan_address()
            where = f" at http://{lan}:{args.ui_port}" if lan else ""
            print(f"Listening on all interfaces: other devices on your network can use the UI{where} and the API.", flush=True)
        if args.open_browser:
            webbrowser.open(ui_url)
        while True:
            check_alive(children)
            time.sleep(0.5)
    except ChildExited as exc:
        print(f"{exc}; stopping the other server.", file=sys.stderr, flush=True)
        exit_code = exc.returncode if exc.returncode > 0 else 1
    except TimeoutError as exc:
        print(f"Startup failed: {exc} (after {args.startup_timeout:.0f} s).", file=sys.stderr, flush=True)
        exit_code = 1
    except (KeyboardInterrupt, StopRequested):
        print("\nStopping the API and the web UI ...", flush=True)
    finally:
        for signum in handled:  # let the shutdown finish
            signal.signal(signum, signal.SIG_IGN)
        stop(children, args.stop_timeout)
        for signum, handler in previous.items():
            if handler is not None:  # None: installed outside Python, cannot be restored
                signal.signal(signum, handler)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
