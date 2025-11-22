#!/bin/bash

sys=$(uname -s)

if [[ $sys == Linux ]]; then
    echo "Sudo password is required to prevent system from sleeping during benchmarks"
    sudo nmcli networking off
    sudo systemd-inhibit python3 scripts/benchmark.py 
    sudo nmcli networking on
elif [[ $sys == Darwin ]]; then
    networksetup -setairportpower en0 off
    caffeinate python3 scripts/benchmark.py
    networksetup -setairportpower en0 on
fi

sleep 20

cd ../project-site 

git pull

python3 update_entries.py --part part2

git add .

git commit -m "Updating benchmarks"

git push
