#!/usr/bin/env bash

# Number of instances you want to run in parallel
n=16
H=4
W=4

# Create a function to run battlesnake instances
run_battlesnake_single() {
    battlesnake play --height $H --width $W -n player1 -u http://localhost:3000
}

run_battlesnake_multi() {
    battlesnake play --height $H --width $W -n player1 -u http://localhost:3000 -n player2 -u http://localhost:3000
}

run_battlesnake() {
    rand=$(($RANDOM%1))
    if [ $rand -eq 0 ]; then
        run_battlesnake_multi
    else
        run_battlesnake_single
    fi
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
