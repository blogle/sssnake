#!/usr/bin/env bash

# Number of instances you want to run in parallel
n=16
H=11
W=11

# Create a function to run battlesnake instances
run_battlesnake_solo() {
    battlesnake play --height $H --width $W -n player1 -u http://localhost:3000
}

run_battlesnake_duel() {
    battlesnake play --height $H --width $W \
        -n player1 -u http://localhost:3000 -n player2 -u http://localhost:3000
}

run_battlesnake_standard() {
    battlesnake play --height $H --width $W \
        -n player1 -u http://localhost:3000 -n player2 -u http://localhost:3000 \
        -n player3 -u http://localhost:3000 -n player4 -u http://localhost:3000 \
        -n player5 -u http://localhost:3000 -n player6 -u http://localhost:3000
}

run_battlesnake() {
    rand=$(($RANDOM%10))
    if [ $rand -lt 2 ]; then
        run_battlesnake_solo
    elif [ $rand -lt 5 ]; then
        run_battlesnake_duel
    else
        run_battlesnake_standard
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
