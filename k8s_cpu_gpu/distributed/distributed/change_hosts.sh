#!/bin/bash
source ./servers.sh

cp hosts allhosts
for i in ${!servers[*]}; do
    echo ${servers[$i]}
    echo -e "${servers[$i]}\t${hostname_prefix}$[$i+1]\t${hostname_prefix}$[$i+1]${hostname_suffix}" | sudo tee -a allhosts >/dev/null 2>&1
done
