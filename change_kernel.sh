
#!/bin/bash
servers=(\
'172.16.0.4' \
'172.16.0.5' \
'172.16.0.6' \
'172.16.0.7' \
'172.16.0.8' \
'172.16.0.9' \
)
 
hostnames=(\
'linux-node1.example.com' \
'linux-node2.example.com' \
'linux-node3.example.com' \
'linux-node4.example.com' \
'linux-node5.example.com' \
'linux-node6.example.com' \
)
 
for i in ${!servers[*]}; do
    echo ${servers[$i]}
    scp k8s.conf root@${servers[$i]}:/etc/sysctl.d/k8s.conf
    ssh -i ~/kp-nzlsmcgq -o StrictHostKeyChecking=no root@${servers[$i]} 'modprobe br_netfilter'
    ssh -i ~/kp-nzlsmcgq -o StrictHostKeyChecking=no root@${servers[$i]} 'sysctl -p /etc/sysctl.d/k8s.conf'
done
