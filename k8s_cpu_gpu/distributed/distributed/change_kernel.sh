
#!/bin/sh
source ./servers.sh
 
for i in ${!servers[*]}; do
    echo ${servers[$i]}
    scp k8s.conf root@${servers[$i]}:/etc/sysctl.d/k8s.conf
    ssh root@${servers[$i]} 'modprobe br_netfilter'
    ssh root@${servers[$i]} 'sysctl -p /etc/sysctl.d/k8s.conf'
done
