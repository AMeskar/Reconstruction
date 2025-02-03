#!/bin/bash
pip3 install six --user
pip3 install ROOT --user

filename=$1
while IFS= read -r line
do
  echo $line
  xrdcp "$line" ./
  tar_file=$(basename "$line")
  tar -xvf "$tar_file"
  rm "$tar_file"
  echo "$tar_file"
done < "$filename"
