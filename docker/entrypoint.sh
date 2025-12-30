#!/bin/bash
set -e
#
# This script starts the D-Bus service and then executes the main container command.
#

# Start a system dbus daemon if available and not already running.
# This prevents "Failed to connect to the bus" errors from Chromium.
if command -v dbus-daemon >/dev/null 2>&1; then
	if [ ! -S /run/dbus/system_bus_socket ]; then
		# Ensure the directory exists and start the daemon
		mkdir -p /run/dbus
		dbus-daemon --system --fork
	fi
fi

# Default Chromium flags for cloud environments (can be overridden by env)
: "${CHROMIUM_FLAGS:--no-sandbox --disable-dev-shm-usage}"
export BROWSER_LAUNCH_ARGS="${CHROMIUM_FLAGS}"

# Ensure Playwright uses the copied browser binaries (match base image)
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/opt/playwright}"

# Default Gunicorn timeout (seconds). Override with env `GUNICORN_TIMEOUT`.
: "${GUNICORN_TIMEOUT:=120}"

# If the command is Gunicorn and no timeout is specified, append one.
if [ "$#" -ge 1 ]; then
	cmd_basename=$(basename "$1")
	if [ "$cmd_basename" = "gunicorn" ] || echo "$1" | grep -q "gunicorn"; then
		has_timeout=false
		for arg in "$@"; do
			case "$arg" in
				--timeout|--graceful-timeout) has_timeout=true; break ;;
				--timeout=*) has_timeout=true; break ;;
			esac
		done
		if [ "$has_timeout" = "false" ]; then
			set -- "$@" --timeout "$GUNICORN_TIMEOUT"
		fi
	fi
fi

# Drop privileges and execute the main command (e.g., gunicorn) as the 'appuser'.
# This is a security best practice.
exec gosu appuser "$@"
