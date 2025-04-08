
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
    scp /etc/apt/sources.list root@${servers[$i]}:/etc/apt/sources.list
    ssh -i ~/kp-nzlsmcgq -o StrictHostKeyChecking=no root@${servers[$i]} 'curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/saltstack/py3/ubuntu/18.04/amd64/latest/SALTSTACK-GPG-KEY.pub | sudo apt-key add -'
    ssh -i ~/kp-nzlsmcgq -o StrictHostKeyChecking=no root@${servers[$i]} 'curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -'
    ssh -i ~/kp-nzlsmcgq -o StrictHostKeyChecking=no root@${servers[$i]} 'apt update'
done
