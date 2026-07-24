#!/usr/bin/env bash
# Install all camera udev rules + apply-scripts from ./udev, so camera settings
# and stable device names persist across reboots and /dev/videoN renumbering.
#   sudo ./scripts/install-camera-udev.sh
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo:  sudo $0" >&2; exit 1; }
root="$(cd "$(dirname "$0")/.." && pwd)"

# apply-scripts -> /usr/local/bin, rules -> /etc/udev/rules.d
for f in "$root"/udev/*.sh;    do [ -e "$f" ] && install -m 0755 "$f" "/usr/local/bin/$(basename "$f")"      && echo "installed /usr/local/bin/$(basename "$f")"; done
for f in "$root"/udev/*.rules; do [ -e "$f" ] && install -m 0644 "$f" "/etc/udev/rules.d/$(basename "$f")" && echo "installed /etc/udev/rules.d/$(basename "$f")"; done

udevadm control --reload-rules
udevadm trigger --subsystem-match=video4linux --action=add
udevadm trigger --subsystem-match=usb --action=add   # for the autosuspend power rules
sleep 2

echo
echo "=== stable camera symlinks ==="
ls -l /dev/cam-* 2>/dev/null || echo "(none yet — replug the camera if missing)"
echo
echo "=== USB Camera control values ==="
byid="/dev/v4l/by-id/usb-CN02KX4NLG0004ABK00_USB_Camera_CN02KX4NLG0004ABK00-video-index0"
[ -e "$byid" ] && v4l2-ctl -d "$byid" --get-ctrl=auto_exposure,exposure_dynamic_framerate,brightness,backlight_compensation
echo "log: /var/log/usbcam-controls.log"
