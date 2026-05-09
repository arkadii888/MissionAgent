#!/usr/bin/env bash
set -euo pipefail

TOTAL_STEPS=10
CURRENT_STEP=0
START_TS="$(date +%s)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Git checkout root (parent of agent/)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"

declare -a PASS_ITEMS=()
declare -a WARN_ITEMS=()
declare -a FAIL_ITEMS=()

_ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

step() {
  CURRENT_STEP=$((CURRENT_STEP + 1))
  echo
  echo "[$(_ts)] [STEP ${CURRENT_STEP}/${TOTAL_STEPS}] $*"
}

info() {
  echo "[$(_ts)] [INFO] $*"
}

ok() {
  echo "[$(_ts)] [OK] $*"
}

warn() {
  echo "[$(_ts)] [WARN] $*"
}

fail() {
  echo "[$(_ts)] [FAIL] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command missing: $1"
    FAIL_ITEMS+=("Missing command: $1")
    exit 1
  fi
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

record_stage_time() {
  local stage_start="$1"
  local stage_name="$2"
  local elapsed
  elapsed="$(( $(date +%s) - stage_start ))"
  info "Stage '${stage_name}' finished in ${elapsed}s"
}

apt_install_with_fallback() {
  local pkg
  local -a candidates=("$@")
  for pkg in "${candidates[@]}"; do
    if run_sudo apt-get install -y "${pkg}"; then
      ok "Installed package: ${pkg}"
      return 0
    fi
    warn "Package unavailable on this image: ${pkg}"
  done
  return 1
}

camera_list_cmd() {
  if command -v rpicam-hello >/dev/null 2>&1; then
    echo "rpicam-hello --list-cameras"
    return 0
  fi
  if command -v libcamera-hello >/dev/null 2>&1; then
    echo "libcamera-hello --list-cameras"
    return 0
  fi
  return 1
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  info "uv not found; installing to ~/.local/bin via astral.sh installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
}

step "Preflight checks and environment summary"
require_cmd python3
require_cmd apt-get
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
  fail "pyproject.toml not found at repo root: ${REPO_ROOT}"
  exit 1
fi
if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
  fail "This script needs root privileges. Re-run as root or install sudo."
  exit 1
fi
if ! command -v uv >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
  info "Installing curl (needed to bootstrap uv)"
  run_sudo apt-get update
  run_sudo apt-get install -y curl
fi
if ! command -v uv >/dev/null 2>&1; then
  require_cmd curl
fi
ensure_uv
require_cmd uv
info "Repo root: ${REPO_ROOT}"
info "Python: $(python3 --version 2>&1)"
if [ -f /etc/os-release ]; then
  info "OS: $(. /etc/os-release && echo "${PRETTY_NAME:-unknown}")"
fi
PASS_ITEMS+=("Preflight checks passed")

step "APT refresh and base tools"
stage_start="$(date +%s)"
run_sudo apt-get update
run_sudo apt-get install -y \
  curl \
  jq \
  pciutils \
  dkms \
  python3-dev \
  pkg-config
# libatlas-base-dev was common on older Raspberry Pi OS; Debian Trixie+ dropped it in favor of OpenBLAS.
if apt_install_with_fallback libatlas-base-dev libopenblas-dev; then
  ok "BLAS dev headers installed (ATLAS or OpenBLAS)"
else
  warn "Could not install libatlas-base-dev or libopenblas-dev; continuing (only needed for some native Python builds)"
fi
ok "Base tools installed"
record_stage_time "${stage_start}" "APT refresh and base tools"
PASS_ITEMS+=("Base tools installed")

step "Install native camera stack (Picamera2/libcamera)"
stage_start="$(date +%s)"
run_sudo apt-get install -y python3-picamera2
if apt_install_with_fallback libcamera-apps rpicam-apps; then
  PASS_ITEMS+=("Camera CLI package installed")
else
  WARN_ITEMS+=("Could not install libcamera/rpicam CLI apps")
fi
ok "Camera Python stack installed"
record_stage_time "${stage_start}" "Camera stack install"
PASS_ITEMS+=("Picamera2 installed")

step "Install project Python runtime (uv sync, repo root)"
stage_start="$(date +%s)"
cd "${REPO_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"
# --system-site-packages: APT packages (python3-picamera2, hailo bindings) live in system site-packages
if [ -d "${VENV_PATH}" ]; then
  info "Using existing venv at ${VENV_PATH} (delete it for a completely clean Python env)"
  uv venv --python 3.13 --system-site-packages --allow-existing "${VENV_PATH}"
else
  uv venv --python 3.13 --system-site-packages "${VENV_PATH}"
fi
uv sync
ok "Project dependencies installed with uv at ${VENV_PATH}"
record_stage_time "${stage_start}" "Python dependencies"
PASS_ITEMS+=("Python dependencies installed (uv sync)")

step "Install Hailo runtime packages (hailo-all path)"
stage_start="$(date +%s)"
if run_sudo apt-get install -y hailo-all; then
  ok "Installed hailo-all package"
  PASS_ITEMS+=("hailo-all installed")
else
  warn "hailo-all install failed. Ensure Hailo APT repository is configured for your image."
  WARN_ITEMS+=("hailo-all package not installed")
fi
record_stage_time "${stage_start}" "Hailo package install"

step "Verify Hailo runtime/driver handshake"
stage_start="$(date +%s)"
if command -v hailortcli >/dev/null 2>&1; then
  if hailortcli fw-control identify >/tmp/hailort_identify.out 2>&1; then
    ok "hailortcli identifies Hailo device and firmware"
    PASS_ITEMS+=("hailortcli fw-control identify passed")
  else
    warn "hailortcli exists but identify failed (often fixed by reboot after install)"
    WARN_ITEMS+=("hailortcli identify failed")
  fi
else
  warn "hailortcli not found after install"
  WARN_ITEMS+=("hailortcli command missing")
fi
record_stage_time "${stage_start}" "Hailo runtime verify"

step "Verify camera enumeration"
stage_start="$(date +%s)"
if cam_cmd="$(camera_list_cmd)"; then
  if [ "${cam_cmd}" = "rpicam-hello --list-cameras" ]; then
    rpicam-hello --list-cameras >/tmp/camera_list.out 2>&1
    cam_rc=$?
  else
    libcamera-hello --list-cameras >/tmp/camera_list.out 2>&1
    cam_rc=$?
  fi
  if [ "${cam_rc}" -eq 0 ]; then
    ok "Camera enumerated with: ${cam_cmd}"
    PASS_ITEMS+=("Camera detected by libcamera/rpicam")
  else
    warn "Camera list command failed: ${cam_cmd}"
    WARN_ITEMS+=("Camera enumeration command failed")
  fi
else
  warn "No camera list CLI found (neither rpicam-hello nor libcamera-hello)"
  WARN_ITEMS+=("No camera CLI tool found")
fi
record_stage_time "${stage_start}" "Camera enumeration"

step "Verify Hailo PCIe visibility"
stage_start="$(date +%s)"
if lspci | grep -qi hailo; then
  ok "Hailo device found on PCIe"
  PASS_ITEMS+=("Hailo PCIe device visible")
else
  warn "Hailo PCIe device not found in lspci output"
  WARN_ITEMS+=("Hailo PCIe device not visible")
fi
if dmesg 2>/dev/null | grep -qi hailo; then
  ok "Kernel log contains Hailo entries"
  PASS_ITEMS+=("Kernel has Hailo messages")
else
  warn "No Hailo entries found in dmesg (may require root or driver not loaded)"
  WARN_ITEMS+=("No Hailo dmesg entries")
fi
record_stage_time "${stage_start}" "Hailo PCIe checks"

step "Picamera2 frame capture smoke test"
stage_start="$(date +%s)"
if "${VENV_PATH}/bin/python" - <<'PY'
from picamera2 import Picamera2
cam = Picamera2()
cfg = cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
cam.configure(cfg)
cam.start()
frame = cam.capture_array("main")
cam.stop()
cam.close()
assert frame is not None and frame.size > 0
print("Frame shape:", frame.shape)
PY
then
  ok "Picamera2 captured a frame successfully"
  PASS_ITEMS+=("Picamera2 capture smoke test passed")
else
  fail "Picamera2 frame capture failed"
  FAIL_ITEMS+=("Picamera2 capture smoke test failed")
fi
record_stage_time "${stage_start}" "Picamera2 smoke test"

step "Hailo Python binding smoke test"
stage_start="$(date +%s)"
if "${VENV_PATH}/bin/python" - <<'PY'
from hailo_platform import VDevice
v = VDevice()
v.release()
print("hailo_platform import and VDevice creation OK")
PY
then
  ok "hailo_platform import and VDevice creation succeeded"
  PASS_ITEMS+=("hailo_platform smoke test passed")
else
  warn "hailo_platform smoke test failed"
  WARN_ITEMS+=("hailo_platform import/VDevice failed")
fi
record_stage_time "${stage_start}" "Hailo Python smoke test"

echo
echo "================ FINAL SUMMARY ================"
echo "PASS ITEMS (${#PASS_ITEMS[@]}):"
for item in "${PASS_ITEMS[@]}"; do
  echo "  - ${item}"
done

echo "WARN ITEMS (${#WARN_ITEMS[@]}):"
if [ "${#WARN_ITEMS[@]}" -eq 0 ]; then
  echo "  - none"
else
  for item in "${WARN_ITEMS[@]}"; do
    echo "  - ${item}"
  done
fi

echo "FAIL ITEMS (${#FAIL_ITEMS[@]}):"
if [ "${#FAIL_ITEMS[@]}" -eq 0 ]; then
  echo "  - none"
else
  for item in "${FAIL_ITEMS[@]}"; do
    echo "  - ${item}"
  done
fi

TOTAL_ELAPSED="$(( $(date +%s) - START_TS ))"
echo "Total elapsed: ${TOTAL_ELAPSED}s"
echo "==============================================="
echo "Remediation hints:"
echo "  - Camera not found or capture fails: check CSI cable orientation, then run 'rpicam-hello --list-cameras'."
echo "  - Hailo not visible on PCIe: reseat Hailo module, confirm PCIe ribbon/hat wiring, reboot, then run 'lspci | grep -i hailo'."
echo "  - Hailo runtime not ready: reboot, then run 'hailortcli fw-control identify' and check it reports HAILO8/HAILO8L."
echo "  - hailo_platform import fails: install Hailo apt repo and rerun 'sudo apt-get install -y hailo-all'."
echo "  - Python deps: from repo root run 'uv sync'; add pytest with 'uv sync --extra dev'; model export tools with 'uv sync --extra export'."
echo "  - Picamera2/Hailo in venv: this script uses 'uv venv --system-site-packages' so APT python3-picamera2 and hailo packages are visible."

if [ "${#FAIL_ITEMS[@]}" -gt 0 ]; then
  fail "Setup completed with failures."
  exit 2
fi

if [ "${#WARN_ITEMS[@]}" -gt 0 ]; then
  warn "Setup completed with warnings. Review WARN ITEMS above."
else
  ok "Setup completed successfully."
fi
