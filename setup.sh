#!/bin/env bash

user="adb321"


exe="cloudlab_depend.sh"

machines=("amd132" "amd145" "amd146" "amd156" "amd157" "amd166" "amd182" "amd191" "amd195" "amd204")
domain="utah.cloudlab.us"

for m in ${machines[@]}; do
  ssh "${user}@$m.$domain" hostname
done

# Send install script 
for m in ${machines[@]}; do
  scp -r "${exe}" "${user}@$m.$domain":~
done

# run install script
for m in ${machines[@]}; do
  ssh "${user}@$m.$domain" bash ${exe}
done