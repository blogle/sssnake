#!/usr/bin/env bash

# Number of instances you want to run in parallel
n=64

# Create a function to run battlesnake instances
run_battlesnake() {
    battlesnake --verbose play -v -s -n 1 -u http://localhost:3000 -u http://localhost:3000
}

# Start the initial set of processes
for ((i=1; i<=n; i++)); do
    run_battlesnake &
done

# Monitor and replace completed processes
while true; do
    # Wait for any background process to complete
    wait -n

    # Start a new process to replace the completed one
    run_battlesnake &
done
