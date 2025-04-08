
#!/bin/sh
source ./servers.sh

for i in ${!servers[*]}; do
    echo ${servers[$i]}
    ssh root@${servers[$i]}  'cp -p /etc/fstab /etc/fstab.bak$(date '+%Y%m%d%H%M%S')'
    ssh root@${servers[$i]} "sed -ri 's/.*swap.*/#&/' /etc/fstab"
    ssh root@${servers[$i]} "swapoff -a"
done
