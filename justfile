set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set ignore-comments := true

default:
    @just --list

rlc2ss +opts:
    uv run scripts/rlc2ss.py {{opts}}

[no-cd]
[windows]
setup build *opts:
    #! powershell
    cmd /c rmdir /s /q {{build}}
    uv venv --allow-existing
    .venv/scripts/activate
    $Env:PKG_CONFIG_PATH="{{build}}\\conan"
    uv run conan install conanfile.txt --build missing --output-folder {{build}}\conan --conf tools.env.virtualenv:powershell=powershell.exe
    # Call conanbuild.ps1 to set the environment variables
    .\\{{build}}\\conan\\conanbuild.ps1
    uv run meson setup -Dcpp_std=c++latest {{build}} {{opts}}

[no-cd]
[linux]
setup build *opts:
    #!/usr/bin/env bash
    rm -rf {{build}}
    export UV_LINK_MODE=copy
    uv venv --allow-existing
    source .venv/bin/activate
    export PKG_CONFIG_PATH="{{build}}/conan"
    uv run conan install conanfile.txt --build missing --output-folder {{build}}/conan
    source ./{{build}}/conan/conanbuild.sh
    uv run meson setup {{build}} {{opts}}

[no-cd]
[windows]
build build_folder="build" *opts="":
    #! powershell
    .venv/scripts/activate
    meson compile -C {{build_folder}}

[no-cd]
[linux]
build build_folder="build" *opts="":
    #!/usr/bin/env bash
    source .venv/bin/activate
    ninja -C {{build_folder}}

[no-cd]
[windows]
test build_folder="build" *opts="":
    #! powershell
    .venv/scripts/activate
    meson compile -C {{build_folder}} tests
    & .\{{build_folder}}\tests.exe {{opts}}

[no-cd]
[linux]
test build_folder="build" *opts="":
    #!/usr/bin/env bash
    source .venv/bin/activate
    ninja -C {{build_folder}} tests
    ./{{build_folder}}/tests {{opts}}
