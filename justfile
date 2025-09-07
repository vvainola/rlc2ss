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
    if (!(Test-Path -Path .venv)) {
        uv venv
    }
    .venv/scripts/activate
    $Env:PKG_CONFIG_PATH="{{build}}\\conan"
    uv run conan install . --build missing --output-folder {{build}}\conan --conf tools.env.virtualenv:powershell=powershell.exe
     # Call conanbuild.ps1 to set the environment variables
    .\\{{build}}\\conan\\conanbuild.ps1
    uv run meson setup {{build}} {{opts}}
    uv run ..\meson-ninja-vs\ninja_vs.py -b {{build}}
