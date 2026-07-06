# Ansible自动化配置管理

## Ansible简介

Ansible是一款开源的IT自动化工具，用于配置管理、应用部署、编排等任务。由Michael DeHaan于2012年创建，后被Red Hat收购。Ansible采用无代理（Agentless）架构，通过SSH协议连接到目标主机执行任务，无需在目标主机上安装任何客户端。

## 核心概念

### 控制节点与被控节点
- **控制节点（Control Node）**：安装Ansible的机器，负责执行任务
- **被控节点（Managed Nodes）**：被管理的目标服务器

### Inventory清单
定义被管理主机的列表：

```ini
# inventory/hosts
[webservers]
web01.example.com ansible_host=192.168.1.10
web02.example.com ansible_host=192.168.1.11

[dbservers]
db01.example.com ansible_host=192.168.1.20

[production:children]
webservers
dbservers
```

### 模块（Modules）
Ansible提供数百个内置模块：

| 模块 | 用途 |
|------|------|
| copy | 复制文件到远程主机 |
| template | 使用Jinja2模板生成文件 |
| yum/apt | 软件包管理 |
| service | 服务管理（启动/停止/重启） |
| file | 文件和目录属性管理 |
| command/shell | 执行命令 |
| git | Git仓库操作 |
| docker_container | Docker容器管理 |

## Playbook编写

Playbook是Ansible的配置、部署和编排语言，使用YAML格式：

```yaml
---
- name: 部署Web应用
  hosts: webservers
  become: yes
  vars:
    app_port: 8080
    app_version: "1.2.0"

  tasks:
    - name: 安装Nginx
      yum:
        name: nginx
        state: present

    - name: 部署配置文件
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: restart nginx

    - name: 启动Nginx服务
      service:
        name: nginx
        state: started
        enabled: yes

  handlers:
    - name: restart nginx
      service:
        name: nginx
        state: restarted
```

### 变量与事实

```yaml
- name: 使用变量示例
  hosts: all
  vars:
    greeting: "Hello"
  tasks:
    - debug:
        msg: "{{ greeting }}, host is {{ ansible_hostname }}"
```

`ansible_hostname` 是Ansible自动收集的事实（Facts）之一。

## Roles角色

Roles是Ansible的组织方式，将变量、任务、模板等按标准目录结构组织：

```
roles/
  webserver/
    tasks/
      main.yml
    handlers/
      main.yml
    templates/
      nginx.conf.j2
    vars/
      main.yml
    defaults/
      main.yml
```

使用Role：
```yaml
- hosts: webservers
  roles:
    - common
    - webserver
```

## Ad-Hoc命令

无需编写Playbook，直接执行单条命令：

```bash
# 检查所有主机连通性
ansible all -m ping

# 在所有web服务器上查看内存使用
ansible webservers -m shell -a "free -h"

# 在所有主机上安装htop
ansible all -m yum -a "name=htop state=present" --become

# 复制文件到所有主机
ansible all -m copy -a "src=/tmp/config dest=/etc/app/config"
```

## Ansible最佳实践

1. **使用Roles组织代码**，提高复用性
2. **将敏感信息放入Ansible Vault**加密存储
3. **使用动态Inventory**对接云平台（AWS/Azure）
4. **Playbook幂等性**，多次执行结果一致
5. **使用ansible-lint检查Playbook规范**
