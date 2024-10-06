set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set ignore-comments := true

setup:
    uv venv

rlc2ss +opts:
    uv run scripts\rlc2ss.py {{opts}}

alias g := generate
[no-cd]
generate build *opts:
    #! powershell
    cmd /c rmdir /s /q {{build}}
    $Env:PKG_CONFIG_PATH="{{build}}\\conan"
    uv run conan install . -g pkg_config --build missing --install-folder {{build}}\conan
    uv run meson setup {{build}} {{opts}}
    uv run ..\meson-ninja-vs\ninja_vs.py -b {{build}}
