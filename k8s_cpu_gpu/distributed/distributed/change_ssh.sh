#!/bin/bash
source ./servers.sh
pass=Tsinghua2020

for i in ${!servers[*]}; do
    expect -c "
    spawn ssh -o StrictHostKeyChecking=no ${servers[$i]} \"echo ${servers[$i]}\"
    expect \"password:\"
    send \"${pass}\r\"
    expect eof
    "
    expect -c "
    spawn ssh-copy-id ${servers[$i]}
    expect \"password:\"
    send \"${pass}\r\"
    expect eof
    "
done

