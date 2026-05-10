#!/usr/bin/env bash
set -euo pipefail

TOTAL_STEPS=11
CURRENT_STEP=0
START_TS="$(date +%s)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"

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

# SKIP_ARDUCAM_PRIVARIETY_INSTALL=1 skips; override URL with ARDUCAM_PIVARIETY_INSTALL_SCRIPT_URL.
install_arducam_pivariety_native_stack() {
  local def_url dl_url installer
  if [ "${SKIP_ARDUCAM_PRIVARIETY_INSTALL:-}" = "1" ]; then
    info "Skipping Arducam install_pivariety_pkgs.sh (SKIP_ARDUCAM_PRIVARIETY_INSTALL=1)"
    return 1
  fi
  def_url="https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh"
  dl_url="${ARDUCAM_PIVARIETY_INSTALL_SCRIPT_URL:-${def_url}}"
  installer="$(mktemp)"

  if command -v curl >/dev/null 2>&1 && curl -fLsso "${installer}" "${dl_url}"; then
    :
  elif command -v wget >/dev/null 2>&1 && wget -qO "${installer}" "${dl_url}"; then
    :
  else
    rm -f "${installer}"
    warn "Could not download install script (curl or wget). Trying APT fallback for camera CLI."
    WARN_ITEMS+=("Arducam installer download failed")
    return 1
  fi

  chmod +x "${installer}"

  for p in libcamera_dev libcamera_apps 64mp_pi_hawk_eye_kernel_driver; do
    info "install_pivariety_pkgs.sh -p ${p}"
    if ! run_sudo bash "${installer}" -p "${p}"; then
      rm -f "${installer}"
      warn "install_pivariety_pkgs.sh failed: ${p}"
      WARN_ITEMS+=("Arducam pivariety -p ${p} failed")
      return 1
    fi
  done

  rm -f "${installer}"
  ok "Arducam libcamera/apps/driver installer finished"
}

camera_list_cmd() {
  local c=""
  if command -v rpicam-still >/dev/null 2>&1; then c="rpicam-still"
  elif command -v rpicam-hello >/dev/null 2>&1; then c="rpicam-hello"
  elif command -v libcamera-still >/dev/null 2>&1; then c="libcamera-still"
  elif command -v libcamera-hello >/dev/null 2>&1; then c="libcamera-hello"
  else return 1
  fi
  echo "${c} --list-cameras"
}

ensure_camera_device_groups() {
  local tgt=""
  [ "$(id -u)" -eq 0 ] && tgt="${SUDO_USER:-}" || tgt="${USER:-}"
  [ -z "${tgt}" ] || [ "${tgt}" = root ] && return 0

  for g in video render; do
    if id -nG "${tgt}" 2>/dev/null | tr ' ' '\n' | grep -qx "${g}"; then
      continue
    fi
    run_sudo usermod -aG "${g}" "${tgt}" && ok "Added ${tgt} to group ${g}" || warn "Could not add ${tgt} to ${g}"
  done
}

ARDUCAM_64MP_CAM1_DT="dtoverlay=arducam-64mp,cam1"

resolve_rpi_boot_config_txt() {
  if run_sudo test -r /boot/firmware/config.txt; then
    printf '%s\n' '/boot/firmware/config.txt'
  elif run_sudo test -r /boot/config.txt; then
    printf '%s\n' '/boot/config.txt'
  fi
}

ensure_camera_auto_detect_disabled() {
  local bc="$1" out=""
  # Sets active camera_auto_detect lines to =0 or appends [all]/camera_auto_detect=0 when missing.
  if ! out="$(run_sudo python3 - "${bc}" <<'PY'
import re
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8", errors="replace") as f:
    raw = f.read()


def lf(s):
    return s.replace("\r\n", "\n").replace("\r", "\n")


def has_uncommented(lines):
    for ln in lines:
        t = ln.lstrip()
        if t.startswith("#"):
            continue
        if re.match(r"camera_auto_detect\s*=", t):
            return True
    return False


lines = lf(raw).split("\n")
out_lines = []

for ln in lines:
    t = ln.lstrip()
    if t.startswith("#"):
        out_lines.append(ln)
        continue
    if re.match(r"camera_auto_detect\s*=", t):
        ws = ln[: len(ln) - len(t)]
        out_lines.append(f"{ws}camera_auto_detect=0")
        continue
    out_lines.append(ln)

if not has_uncommented(out_lines):
    out_lines.extend(
        [
            "",
            "# setup_pi_hailo_arducam: third-party CSI",
            "[all]",
            "camera_auto_detect=0",
        ]
    )

new_txt = lf("\n".join(out_lines) + "\n")
old_txt = lf(raw)
old_canon = old_txt if old_txt.endswith("\n") else (old_txt + "\n" if old_txt else "")

print("CHANGED" if new_txt != old_canon else "UNCHANGED", end="\n")

if new_txt != old_canon:
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(new_txt)
PY
  )"; then
    warn "Could not patch camera_auto_detect in ${bc}"
    WARN_ITEMS+=("camera_auto_detect firmware edit failed")
    return 0
  fi
  case "${out}" in
    CHANGED*)
      ok "Firmware ${bc}: camera_auto_detect forced off for third-party camera"
      ;;
    UNCHANGED*)
      ok "Firmware ${bc}: camera_auto_detect already ok"
      ;;
    *)
      warn "Unexpected firmware helper output."
      WARN_ITEMS+=("camera_auto_detect tooling output ambiguous")
      ;;
  esac
}

ensure_arducam64mp_cam1_dtoverlay() {
  local bc="$1"

  if run_sudo grep -qFx "${ARDUCAM_64MP_CAM1_DT}" "${bc}"; then
    ok "${bc} already has ${ARDUCAM_64MP_CAM1_DT}"
    return 0
  fi

  if run_sudo grep -qF "arducam-64mp" "${bc}"; then
    warn "${bc} references arducam-64mp with a different option than '${ARDUCAM_64MP_CAM1_DT}'. Edit manually."
    WARN_ITEMS+=("config.txt arducam line mismatch")
    return 0
  fi

  {
    printf '%s\n' "" "# setup_pi_hailo_arducam: Arducam 64MP CAM1" "[all]" "${ARDUCAM_64MP_CAM1_DT}"
  } | run_sudo tee -a "${bc}" >/dev/null
  ok "Appended ${ARDUCAM_64MP_CAM1_DT} to ${bc} (reboot to apply firmware)"
}

ensure_uv() {
  command -v uv >/dev/null 2>&1 && return 0
  info "Installing uv (~/.local/bin)"
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
  fail "This script needs sudo or root."
  exit 1
fi
if ! command -v uv >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
  run_sudo apt-get update && run_sudo apt-get install -y curl
fi
! command -v uv >/dev/null 2>&1 && require_cmd curl

ensure_uv
require_cmd uv
info "Repo root: ${REPO_ROOT}  Python: $(python3 --version 2>&1)"
[ -f /etc/os-release ] && info "$(. /etc/os-release && echo "${PRETTY_NAME:-unknown}")"

step "APT refresh and base tools"
run_sudo apt-get update
run_sudo apt-get install -y \
  curl jq pciutils dkms python3-dev pkg-config

apt_install_with_fallback libatlas-base-dev libopenblas-dev || warn "Optional BLAS dev headers not installed"

step "Firmware config (camera_auto_detect + Arducam overlay)"
_bc="$(resolve_rpi_boot_config_txt)"
if [ -z "${_bc}" ]; then
  warn "No /boot/.../config.txt readable; skipping firmware edits."
  WARN_ITEMS+=("boot config.txt unavailable")
else
  ensure_camera_auto_detect_disabled "${_bc}"
  ensure_arducam64mp_cam1_dtoverlay "${_bc}"
fi
unset _bc

step "Camera stack (Arducam pivariety or APT fallback + Picamera2)"
if install_arducam_pivariety_native_stack; then
  ok "Arducam install_pivariety_pkgs steps completed"
else
  apt_install_with_fallback libcamera-apps rpicam-apps || WARN_ITEMS+=("No stock libcamera-apps/rpicam-apps")
fi
run_sudo apt-get install -y python3-picamera2
ensure_camera_device_groups

step "Python project deps (uv sync)"
cd "${REPO_ROOT}"
export PATH="${HOME}/.local/bin}:${PATH}"

if [ -d "${VENV_PATH}" ]; then
  uv venv --python 3.13 --system-site-packages --allow-existing "${VENV_PATH}"
else
  uv venv --python 3.13 --system-site-packages "${VENV_PATH}"
fi
uv sync
ok "uv sync done at ${VENV_PATH}"

step "Hailo packages (hailo-all)"
run_sudo apt-get install -y hailo-all \
  || WARN_ITEMS+=("hailo-all not installed")

step "Verify Hailo (hailortcli)"
if command -v hailortcli >/dev/null 2>&1; then
  hailortcli fw-control identify >/tmp/hailort_identify.out 2>&1 \
    || WARN_ITEMS+=("hailortcli identify failed; try reboot")
else
  warn "hailortcli missing"
  WARN_ITEMS+=("hailortcli missing")
fi

step "Camera CLI list test"
if cam_cmd="$(camera_list_cmd)"; then
  bash -c "${cam_cmd}" >/tmp/camera_list.out 2>&1 \
    || WARN_ITEMS+=("camera list CLI failed: ${cam_cmd}")
else
  warn "No rpicam*/libcamera* CLI"
  WARN_ITEMS+=("no camera CLI in PATH")
fi

step "Hailo PCIe / dmesg"
lspci 2>/dev/null | grep -qi hailo || WARN_ITEMS+=("no Hailo in lspci")
dmesg 2>/dev/null | grep -qi hailo || WARN_ITEMS+=("no Hailo in dmesg")

step "Picamera2 smoke capture"
smoke="$(mktemp)"
trap 'rm -f "${smoke}"' EXIT

cat >"${smoke}" <<'PY'
import sys
import time
from picamera2 import Picamera2

infos = []
for _ in range(3):
    infos = Picamera2.global_camera_info()
    if infos:
        break
    time.sleep(1.0)

if not infos:
    print("Picamera2: no cameras in global_camera_info()", file=sys.stderr)
    sys.exit(1)

cam = Picamera2(camera_num=0)
cfg = cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
cam.configure(cfg)
cam.start()
frame = cam.capture_array("main")
cam.stop()
cam.close()
assert frame is not None and frame.size > 0
print("Picamera2 frame:", frame.shape)
PY

if "${VENV_PATH}/bin/python" "${smoke}"; then
  ok "Picamera2 capture ok"
elif [ "$(id -u)" -ne 0 ] && command -v sg >/dev/null 2>&1 \
  && sg video -c "$(printf '%q %q' "${VENV_PATH}/bin/python" "${smoke}")"
then
  ok "Picamera2 capture ok (sg video)"
else
  fail "Picamera2 capture failed"
  FAIL_ITEMS+=("Picamera2 smoke test failed")
fi
rm -f "${smoke}"
trap - EXIT

step "Hailo Python (hailo_platform)"
"${VENV_PATH}/bin/python" - <<'PY' || WARN_ITEMS+=("hailo_platform smoke failed")
from hailo_platform import VDevice
v = VDevice()
v.release()
print("hailo_platform OK")
PY

echo
echo "================ SUMMARY ================="
TOTAL_ELAPSED="$(( $(date +%s) - START_TS ))"
echo "Elapsed: ${TOTAL_ELAPSED}s"

if [ "${#WARN_ITEMS[@]}" -gt 0 ]; then
  echo "Warnings (${#WARN_ITEMS[@]}):"
  for item in "${WARN_ITEMS[@]}"; do
    echo "  - ${item}"
  done
else
  echo "Warnings: none"
fi

if [ "${#FAIL_ITEMS[@]}" -gt 0 ]; then
  echo "Failures (${#FAIL_ITEMS[@]}):"
  for item in "${FAIL_ITEMS[@]}"; do
    echo "  - ${item}"
  done
  echo
  echo "Tips: firmware edits need reboot; Arducam: https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/64MP-Hawkeye/"
  fail "Finished with failures."
  exit 2
fi

if [ "${#WARN_ITEMS[@]}" -gt 0 ]; then
  warn "Finished with warnings (see above)."
else
  ok "Finished successfully."
fi
