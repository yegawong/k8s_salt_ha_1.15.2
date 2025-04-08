#!/bin/sh
name_pre=node

for ((i=1; i<=100; i ++))
do
    echo $name_pre$i
    scp -i /root/.ssh/id_rsa -o StrictHostKeyChecking=no ./tmp root@$name_pre$i:/path
    ssh -i /root/.ssh/id_rsa -o StrictHostKeyChecking=no root@$name_pre$i 'command'
done
