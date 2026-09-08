"""Install the refresh as a real scheduled job.

macOS uses launchd, not cron: a LaunchAgent survives reboots and logouts, and
launchd runs a calendar job it missed while the machine was asleep — which cron
does not, and a laptop is asleep most nights. `--print-only` emits the unit for
Linux (systemd or crontab) instead.
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "co.interiorcatalog.refresh"
AGENT_DIR = Path.home() / "Library" / "LaunchAgents"


def plist_path() -> Path:
    return AGENT_DIR / f"{LABEL}.plist"


def _job(project: Path, hour: int, minute: int, extra: list[str]) -> dict:
    logs = project / "data" / "logs"
    return {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "catalog", "refresh", *extra],
        "WorkingDirectory": str(project),
        "EnvironmentVariables": {"PYTHONPATH": str(project)},
        "StartCalendarInterval": [{"Hour": hour, "Minute": minute}],
        "StandardOutPath": str(logs / "refresh.log"),
        "StandardErrorPath": str(logs / "refresh.err.log"),
        "RunAtLoad": False,
        # A crawl is I/O bound and should never compete with the user's own work.
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "Nice": 5,
    }


def render(project: Path, hour: int, minute: int, extra: list[str]) -> bytes:
    return plistlib.dumps(_job(project, hour, minute, extra), sort_keys=False)


def install(project: Path, hour: int, minute: int, extra: list[str], print_only: bool = False) -> int:
    project = Path(project).resolve()
    body = render(project, hour, minute, extra)

    if print_only or sys.platform != "darwin":
        print(_non_macos_help(project, hour, minute, extra) if sys.platform != "darwin"
              else body.decode())
        return 0

    (project / "data" / "logs").mkdir(parents=True, exist_ok=True)
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    path = plist_path()
    path.write_bytes(body)

    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LABEL}"],
                   capture_output=True)          # ignore "not loaded"
    done = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(path)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        # Older macOS releases only understand load -w.
        done = subprocess.run(["launchctl", "load", "-w", str(path)],
                              capture_output=True, text=True)
    if done.returncode != 0:
        print(f"could not load the agent: {done.stderr.strip() or done.stdout.strip()}")
        print(f"the plist is written at {path}; load it yourself with:\n"
              f"  launchctl bootstrap gui/{uid} {path}")
        return 1

    print(f"scheduled: refresh runs daily at {hour:02d}:{minute:02d}")
    print(f"  agent : {path}")
    print(f"  logs  : {project / 'data' / 'logs' / 'refresh.log'}")
    print(f"  run now: launchctl kickstart -p gui/{uid}/{LABEL}")
    print(f"  remove : python3 -m catalog schedule uninstall")
    return 0


def uninstall() -> int:
    path = plist_path()
    if sys.platform == "darwin":
        uid = os.getuid()
        subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LABEL}"], capture_output=True)
        subprocess.run(["launchctl", "unload", "-w", str(path)], capture_output=True)
    if path.exists():
        path.unlink()
        print(f"removed {path}")
    else:
        print("nothing scheduled")
    return 0


def status(db_path: str) -> int:
    path = plist_path()
    print(f"agent file : {path}  ({'present' if path.exists() else 'not installed'})")
    if sys.platform == "darwin":
        out = subprocess.run(["launchctl", "list", LABEL], capture_output=True, text=True)
        print("launchd    : " + ("loaded" if out.returncode == 0 else "not loaded"))
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if any(k in line for k in ('"PID"', '"LastExitStatus"')):
                    print("             " + line.strip().rstrip(";"))

    from .store import connect
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT id, source, started_at, finished_at, status, products_new, "
        "products_changed, price_changes, error FROM crawl_runs "
        "ORDER BY started_at DESC LIMIT 10"
    ).fetchall()
    if not rows:
        print("\nno refresh has run yet")
        return 0
    print(f"\n{'run':>4}  {'started':<20} {'status':<8} {'new':>5} {'chg':>5} {'price':>6}  source")
    for r in rows:
        print(f"{r['id']:>4}  {r['started_at'][:19].replace('T', ' '):<20} "
              f"{r['status']:<8} {r['products_new']:>5} {r['products_changed']:>5} "
              f"{r['price_changes']:>6}  {r['source']}"
              + (f"\n      {r['error']}" if r["error"] else ""))
    return 0


def _non_macos_help(project: Path, hour: int, minute: int, extra: list[str]) -> str:
    cmd = f"{sys.executable} -m catalog refresh " + " ".join(extra)
    return f"""# systemd user timer — ~/.config/systemd/user/interior-catalog.service
[Unit]
Description=Interior catalog refresh

[Service]
Type=oneshot
WorkingDirectory={project}
ExecStart={cmd}

# ~/.config/systemd/user/interior-catalog.timer
[Unit]
Description=Daily interior catalog refresh

[Timer]
OnCalendar=*-*-* {hour:02d}:{minute:02d}:00
Persistent=true

[Install]
WantedBy=timers.target

# then:  systemctl --user enable --now interior-catalog.timer
#
# or crontab -e:
# {minute} {hour} * * * cd {project} && {cmd} >> data/logs/refresh.log 2>&1
"""
