@echo off
cls
setlocal enabledelayedexpansion
rmdir /s /q %1
set PKG_CONFIG_PATH=%1\conan
conan install . -g pkg_config --build missing --install-folder %1\conan
meson setup %*
