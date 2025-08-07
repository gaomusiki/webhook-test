#!/bin/bash

control_file="/tmp/loop_test_script_control"
SEP="========================================================================"

touch $control_file

script="test_script.sh"

interval=7200 # 2 hours
# interval=3600 # 1 hour
# interval=1800 # 30 min
# interval=600 # 10 min

cnt=0
while [ -f "$control_file" ]
do
  start_time=$(date +%s)

  echo "$SEP"
  echo "Running the $script for the $((cnt++))-th time. Starting date: $(date)"
  echo "$SEP"

  /bin/bash $script

  end_time=$(date +%s)
  duration=$((end_time - start_time))
  sleep_time=$((interval - duration))

  if [ $sleep_time -gt 0 ]; then
    sleep $sleep_time
  fi
done
