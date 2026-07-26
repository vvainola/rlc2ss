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
    uv run conan install conanfile-windows.txt --build missing --output-folder {{build}}\conan --conf tools.env.virtualenv:powershell=powershell.exe
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
    export PKG_CONFIG_PATH="$(pwd)/{{build}}/conan:$(pkg-config --variable pc_path pkg-config)"
    uv run conan install conanfile-linux.txt --build missing --output-folder {{build}}/conan
    source ./{{build}}/conan/conanbuild.sh
    uv run meson setup {{build}} {{opts}} -Dcpp_std=c++23

[no-cd]
[windows]
build *args:
    #! powershell
    .venv/scripts/activate
    # Use "build" by default; an existing Meson build folder may be given before the build arguments.
    $buildArgs = @("{{args}}" -split " " | Where-Object { $_ })
    $buildFolder = "build"
    if ($buildArgs.Count -gt 0 -and (Test-Path (Join-Path $buildArgs[0] "build.ninja"))) {
        $buildFolder = $buildArgs[0]
        $buildArgs = @($buildArgs | Select-Object -Skip 1)
    }
    meson compile -C $buildFolder @buildArgs

[no-cd]
[linux]
build *args:
    #!/usr/bin/env bash
    # Use "build" by default; an existing Meson build folder may be given before the build arguments.
    read -r -a args <<< "{{args}}"
    build_folder=build
    if (( ${#args[@]} > 0 )) && [[ -f ${args[0]}/build.ninja ]]; then
        build_folder=${args[0]}
        args=("${args[@]:1}")
    fi
    source .venv/bin/activate
    export PKG_CONFIG_PATH="$(pwd)/${build_folder}/conan:$(pkg-config --variable pc_path pkg-config)"
    ninja -C "${build_folder}" "${args[@]}"

[no-cd]
[windows]
test *args:
    #! powershell
    .venv/scripts/activate
    # Use "build" by default; an existing Meson build folder may be given before the test arguments.
    $testArgs = @("{{args}}" -split " " | Where-Object { $_ })
    $buildFolder = "build"
    if ($testArgs.Count -gt 0 -and (Test-Path (Join-Path $testArgs[0] "build.ninja"))) {
        $buildFolder = $testArgs[0]
        $testArgs = @($testArgs | Select-Object -Skip 1)
    }
    # Send job-count options to the build; pass filters and other options to the test executable.
    $buildArgs = @()
    $runnerArgs = @()
    for ($argIndex = 0; $argIndex -lt $testArgs.Count; ++$argIndex) {
        $arg = $testArgs[$argIndex]
        if ($arg -eq "-j" -or $arg -eq "--jobs") {
            $buildArgs += $arg
            if (++$argIndex -lt $testArgs.Count) {
                $buildArgs += $testArgs[$argIndex]
            }
        } elseif ($arg -match "^-j\d+$" -or $arg -match "^--jobs=") {
            $buildArgs += $arg
        } else {
            $runnerArgs += $arg
        }
    }
    meson compile -C $buildFolder tests @buildArgs
    & ".\$buildFolder\tests.exe" @runnerArgs

[no-cd]
[linux]
test *args:
    #!/usr/bin/env bash
    # Use "build" by default; an existing Meson build folder may be given before the test arguments.
    read -r -a args <<< "{{args}}"
    build_folder=build
    if (( ${#args[@]} > 0 )) && [[ -f ${args[0]}/build.ninja ]]; then
        build_folder=${args[0]}
        args=("${args[@]:1}")
    fi
    # Send job-count options to Ninja; pass filters and other options to the test executable.
    build_args=()
    test_args=()
    for ((arg_index = 0; arg_index < ${#args[@]}; ++arg_index)); do
        arg=${args[arg_index]}
        if [[ $arg == -j || $arg == --jobs ]]; then
            build_args+=("$arg")
            if ((++arg_index < ${#args[@]})); then
                build_args+=("${args[arg_index]}")
            fi
        elif [[ $arg == -j[0-9]* || $arg == --jobs=* ]]; then
            build_args+=("$arg")
        else
            test_args+=("$arg")
        fi
    done
    source .venv/bin/activate
    export PKG_CONFIG_PATH="$(pwd)/${build_folder}/conan:$(pkg-config --variable pc_path pkg-config)"
    ninja -C "${build_folder}" tests "${build_args[@]}"
    "./${build_folder}/tests" "${test_args[@]}"
