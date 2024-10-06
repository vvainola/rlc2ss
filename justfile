set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

setup:
    uv venv

rlc2ss +opts:
    uv run scripts\rlc2ss.py {{opts}}
