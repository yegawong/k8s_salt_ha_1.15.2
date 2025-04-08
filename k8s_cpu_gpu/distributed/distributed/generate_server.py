import csv
csv_file = open('./servers.csv', 'r')
r = csv.reader(csv_file)
ips = [i for t in r for i in t]

master=3

with open("./tmproster", "w") as f:
    f.write("# -*- coding: utf-8 -*-\n\
#******************************************\n\
# Author:       Jason Zhao\n\
# Email:        shundong.zhao@linuxhot.com\n\
# Organization: http://www.devopsedu.com/\n\
# Description:  Salt SSH Roster\n\
#******************************************\n\n")
    for i, ip in enumerate(ips):
        f.write("linux-node%d:\n\
host: %s\n\
user: root\n\
priv: /root/.ssh/id_rsa\n\
minion_opts: \n"%(i+1,ip))
        if master > 0:
            f.write("      k8s-role-master: master\n")
            f.write("      etcd-role: node\n")
            f.write("      etcd-name: etcd-node%d\n"%(i+1))
            master-=1
        else:
            f.write("      k8s-role-node: node\n")
        f.write("\n")

with open("./tmpserver.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("servers=(\\\n")
    for ip in ips:
        f.write("'%s' \\\n"%(ip))
    f.write(")\n\n")
    f.write("hostname_prefix='linux-node'\n")
    f.write("hostname_suffix='.example.com'\n")