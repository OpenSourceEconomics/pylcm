#!/usr/bin/env bash
# Take the machine-wide benchmark GPU lock and hold it until the job ends.
#
# usage: hold-benchmark-gpu.sh main|pr
#
# The holder is a background `flock` whose child sleeps forever. Its PID goes to
# `$GITHUB_ENV` as `GPU_LOCK_PID` as soon as it starts, and the job's last step
# kills it, so a job cancelled while waiting does not leave a holder behind.
#
# Main-branch runs go first. A waiting `main` job writes the PID of its blocked
# `flock` to a marker file next to the lock, and a `pr` job does not take the
# lock while that process is alive. A `pr` job that wins a race against a main
# job that has just started waiting hands the lock back before it starts work.
# A running job is never interrupted, so main waits for at most the one
# pull-request benchmark already in progress.
set -euo pipefail

mode=${1:-}
lock=${GPU_BENCHMARK_LOCK:-/tmp/pylcm-gpu-benchmark.lock}
marker=$lock.main-waiting
held=$RUNNER_TEMP/gpu-lock-held
poll=${GPU_BENCHMARK_POLL_SECONDS:-30}

main_is_waiting() {
  local pid
  pid=$(cat "$marker" 2> /dev/null) || return 1
  test -n "$pid" && kill -0 "$pid" 2> /dev/null
}

# Start a lock holder with the given extra `flock` options and print its PID.
start_holder() {
  rm -f "$held"
  nohup flock "$@" "$lock" sh -c 'touch "$0" && exec sleep infinity' "$held" \
    > /dev/null 2>&1 &
  echo "GPU_LOCK_PID=$!" >> "$GITHUB_ENV"
  echo $!
}

stop_holder() {
  pkill -P "$1" 2> /dev/null || true
  kill "$1" 2> /dev/null || true
}

case $mode in
  main)
    pid=$(start_holder)
    echo "$pid" > "$marker"
    until test -e "$held"; do
      echo "$(date): a pull-request benchmark holds the GPU, retrying in ${poll}s..."
      sleep "$poll"
    done
    rm -f "$marker"
    ;;
  pr)
    while :; do
      if main_is_waiting; then
        echo "$(date): a main-branch benchmark waits for the GPU, yielding for ${poll}s..."
        sleep "$poll"
        continue
      fi
      pid=$(start_holder -n)
      # `flock -n` exits at once when the lock is taken; otherwise wait for the marker.
      while ! test -e "$held" && kill -0 "$pid" 2> /dev/null; do sleep 1; done
      if test -e "$held"; then
        main_is_waiting || break
        stop_holder "$pid"
        continue
      fi
      echo "$(date): another benchmark job holds the GPU, retrying in ${poll}s..."
      sleep "$poll"
    done
    ;;
  *)
    echo "usage: $0 main|pr" >&2
    exit 2
    ;;
esac

echo "GPU lock held."
