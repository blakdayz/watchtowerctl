#!/bin/bash

# Check if launchctl is available
if ! command -v launchctl &> /dev/null; then
  echo "launchctl could not be found"
  exit 1
fi

# Get the list of services from launchctl print file
output=$(launchctl print file 2>&1)

# Extract the suggested targets from the output
suggestions=$(echo "$output" | sed -n '/did you mean/,/Usage:/p' | grep -E 'system/|pid/|gui/' | tr '\n' ' ')

# Unload each suggested target
for target in $suggestions; do
  echo "Unloading: $target"
  launchctl unload "$target" 2>/dev/null || echo "Failed to unload $target"
done
