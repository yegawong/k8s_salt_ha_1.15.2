
#!/bin/sh
source ./servers.sh

for i in ${!servers[*]}; do
    echo ${servers[$i]}
    ssh root@${servers[$i]} 'cp /etc/hosts /etc/hosts.bak && cp /etc/hostname /etc/hostname.bak'
    scp allhosts root@${servers[$i]}:/etc/hosts
    ssh root@${servers[$i]} 'echo "'${hostname_prefix}$[$i+1]${hostname_suffix}'" | sudo tee /etc/hostname >/dev/null 2>&1; sudo hostname -F /etc/hostname'
done
