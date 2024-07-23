#!/bin/bash

LOGFILE="./temp/output.log"
INTERVAL=10

> "$LOGFILE"

while true; do
    python engine/server.py 2>&1 | tee -a "$LOGFILE"

    sleep "$INTERVAL"

    : > "$LOGFILE"
done
