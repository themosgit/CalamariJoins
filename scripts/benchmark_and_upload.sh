#!/bin/bash

sys=$(uname -s)

if [[ $sys == Linux ]]; then
    echo "Sudo password is required to prevent system from sleeping during benchmarks"
    sudo systemd-inhibit python3 scripts/benchmark.py 
elif [[ $sys == Darwin ]]; then
    caffeinate &
    python3 scripts/benchmark.py
    pkill caffeinate
fi

cd ../project-site 

git pull

python3 update_entries.py

git add .

git commit -m "updating benchmarks"

git push
