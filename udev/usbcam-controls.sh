#!/usr/bin/env bash
# Apply fixed v4l2 controls to a USB camera capture node.
# Invoked by udev (see 99-usbcam-controls.rules) via `systemd-run --no-block`,
# so it can retry without udev killing it on event completion.
#   $1 = capture node, e.g. /dev/video6
set -u
dev="${1:-}"
log="/var/log/usbcam-controls.log"
[ -n "$dev" ] || { echo "$(date '+%F %T') no device arg" >>"$log" 2>&1; exit 0; }

# Retry: right after enumeration the control interface can briefly be busy.
for i in $(seq 1 5); do
  if /usr/bin/v4l2-ctl -d "$dev" \
        -c auto_exposure=3 \
        -c exposure_dynamic_framerate=0 \
        -c brightness=32 \
        -c backlight_compensation=0 \
        -c gamma=100 \
        -c gain=0 >>"$log" 2>&1; then
    echo "$(date '+%F %T') applied to $dev (attempt $i)" >>"$log" 2>&1
    exit 0
  fi
  sleep 1
done
echo "$(date '+%F %T') FAILED on $dev after 5 attempts" >>"$log" 2>&1
exit 1
