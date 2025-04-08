
#!/bin/bash
servers=(\
'172.16.0.4' \
'172.16.0.5' \
'172.16.0.6' \
'172.16.0.7' \
'172.16.0.8' \
'172.16.0.9' \
)


hosts=(\
'linux-node1' \
'linux-node2' \
'linux-node3' \
'linux-node4' \
'linux-node5' \
'linux-node6' \
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
    ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no root@${servers[$i]} 'cp /etc/hosts /etc/hosts.bak && cp /etc/hostname /etc/hostname.bak'
    ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no root@${servers[$i]} 'echo "'${hostnames[$i]}'" | sudo tee /etc/hostname >/dev/null 2>&1; sudo hostname -F /etc/hostname'
    ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no root@${servers[$i]} 'echo -e "'${servers[$i]}'\t'${hosts[$i]}'\t'${hostnames[$i]}'" | sudo tee -a /etc/hosts >/dev/null 2>&1'
done
