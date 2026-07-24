#!/usr/bin/env bash
# Pre-flight health check before a recording session.
# Verifies each camera actually delivers a frame and flags recent USB instability,
# so you don't start a long record only to have a flaky camera drop on episode 0.
set -u
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

# camera name -> stable by-id capture node (survives /dev/videoN renumbering)
front="/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_01010650-video-index0"
grip="/dev/v4l/by-id/usb-CN02KX4NLG0004ABK00_USB_Camera_CN02KX4NLG0004ABK00-video-index0"
side="/dev/v4l/by-id/usb-046d_Brio_100_2520APYCH8Q8-video-index0"

names=(front grip)
[ -e "$side" ] && names+=(side)   # only check side if it's plugged in

ok=1
echo "=== cameras ==="
for name in "${names[@]}"; do
  path="${!name}"
  node=$(readlink -f "$path" 2>/dev/null)
  if [ -z "$node" ] || [ ! -e "$node" ]; then
    printf "  ❌ %-6s not connected\n" "$name"; ok=0; continue
  fi
  if timeout 12 ffmpeg -hide_banner -loglevel error -f v4l2 -input_format mjpeg \
       -video_size "${CAM_WIDTH}x${CAM_HEIGHT}" -i "$node" -frames:v 1 -f null - >/dev/null 2>&1; then
    printf "  ✅ %-6s streaming (%s)\n" "$name" "$node"
  else
    printf "  ❌ %-6s connected but NOT delivering frames (%s)\n" "$name" "$node"; ok=0
  fi
done

echo "=== USB stability (last 3 min) ==="
log=$( (journalctl -k --since "-3min" 2>/dev/null || dmesg) )
serious=$(echo "$log" | grep -icE "USB disconnect|error -71|error -19")
resets=$(echo  "$log" | grep -icE "reset high-speed")
echo "  disconnects/enumeration errors: $serious"
echo "  bus resets:                     $resets"

echo
if [ "$ok" = 1 ] && [ "$serious" -eq 0 ]; then
  [ "$resets" -gt 0 ] && echo "  (note: $resets bus reset(s) — usually the C270 recovering; not fatal)"
  echo "🟢 GO — cameras healthy, bus quiet enough. Safe to record."
  exit 0
fi
echo "🔴 NO-GO — a camera isn't streaming or the bus just dropped."
echo "   Wait ~10s for the camera to settle, then re-run: ./robot doctor"
exit 1
