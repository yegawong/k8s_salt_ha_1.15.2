#******************************************
# Author:       Jason Zhao
# Email:        shundong.zhao@linuxhot.com
# Organization: http://www.devopsedu.com/
# Description:  Docker Install
#******************************************

include:
  - k8s.modules.base-dir
docker-install:
  cmd.run:
    - name: curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
#  file.managed:
#    - name: /etc/yum.repos.d/docker-ce.repo
#    - source: salt://k8s/templates/docker/docker-ce.repo.template
#    - user: root
#    - group: root
#    - mode: 644
  pkg.installed:
    - name: docker-ce

docker-config:
  file.managed:
    - name: /opt/kubernetes/cfg/docker
    - source: salt://k8s/templates/docker/docker-config.template
    - user: root
    - group: root
    - mode: 644

docker-config-dir:
  file.directory:
    - name: /etc/docker
    
docker-daemon-config:
  file.managed:
    - name: /etc/docker/daemon.json
    - source: salt://k8s/templates/docker/daemon.json.template
    - user: root
    - group: root
    - mode: 644

docker-service:
  file.managed:
    - name: /lib/systemd/system/docker.service
    - source: salt://k8s/templates/docker/docker.service.template
    - user: root
    - group: root
    - mode: 755
  cmd.run:
    - name: systemctl daemon-reload
  service.running:
    - name: docker
    - enable: True
    - watch:
      - file: docker-config
      - file: docker-daemon-config
