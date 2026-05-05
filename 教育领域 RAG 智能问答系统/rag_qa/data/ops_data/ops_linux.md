# Linux 系统管理基础

## 一、Linux 系统概述

Linux 是目前运维领域最核心的操作系统，广泛应用于服务器、云计算、嵌入式等场景。Linux 内核由 Linus Torvalds 于 1991 年发布，遵循 GPL 开源协议。常见的 Linux 发行版包括：

| 发行版 | 包管理器 | 适用场景 |
|--------|----------|----------|
| CentOS / RHEL | yum / dnf | 企业服务器 |
| Ubuntu / Debian | apt / dpkg | 开发环境、云主机 |
| Rocky Linux | dnf | CentOS 替代方案 |
| Alpine Linux | apk | Docker 容器镜像 |

## 二、文件系统与文件操作命令

Linux 遵循"一切皆文件"的哲学，文件系统采用树形层次结构，根目录为 `/`。

### 2.1 核心目录结构

```
/           # 根目录
├── bin/    # 基本命令二进制文件
├── etc/    # 系统配置文件
├── home/   # 用户主目录
├── var/    # 可变数据（日志、缓存）
├── tmp/    # 临时文件
├── usr/    # 用户程序和数据
├── proc/   # 进程和内核信息虚拟文件系统
└── dev/    # 设备文件
```

### 2.2 常用文件操作命令

```bash
# 列出文件详情（包括隐藏文件）
ls -la /var/log

# 查看文件内容
cat /etc/hosts           # 全部输出
less /var/log/messages   # 分页浏览
head -n 20 app.log       # 查看前20行
tail -f /var/log/nginx/access.log  # 实时跟踪日志

# 查找文件
find / -name "nginx.conf" 2>/dev/null
find /var/log -mtime -7 -name "*.log"  # 最近7天修改的日志

# 文件权限管理
chmod 755 script.sh       # rwxr-xr-x
chown nginx:nginx /var/www/html
chmod +x deploy.sh        # 添加执行权限
```

## 三、进程管理命令

进程管理是运维工作的基础，需要掌握进程的查看、控制和优先级调整。

```bash
# 查看进程
ps aux                     # 列出所有进程
ps -ef | grep nginx        # 查找nginx进程
pgrep -f "python app.py"   # 按名称查找PID

# 实时进程监控
top                        # 交互式进程查看器
htop                       # 增强版（需安装）

# 终止进程
kill -9 <PID>              # 强制终止（SIGKILL）
kill -15 <PID>             # 优雅终止（SIGTERM）
pkill -f "gunicorn"        # 按名称终止
killall nginx              # 终止同名进程

# 后台运行程序
nohup python app.py > app.log 2>&1 &
```

### top 命令详解

```
top - 14:30:05 up 30 days,  2:15,  3 users,  load average: 0.52, 0.38, 0.41
Tasks: 156 total,   1 running, 155 sleeping,   0 stopped,   0 zombie
%Cpu(s):  5.2 us,  2.1 sy,  0.0 ni, 92.5 id,  0.1 wa,  0.0 hi,  0.1 si,  0.0 st
MiB Mem :  15896.5 total,   2341.2 free,   8912.3 used,   4643.0 buff/cache
MiB Swap:   4096.0 total,   3890.1 free,    205.9 used.   6023.1 avail Mem

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
 1234 nginx     20   0  452156  32456   1224 S   0.3   0.2   0:05.12 nginx
```

关键指标说明：
- **load average**：1分钟、5分钟、15分钟的平均负载，数值超过CPU核心数表示过载
- **us/sy/wa**：用户态CPU、内核态CPU、IO等待百分比
- **buff/cache**：Linux会用空闲内存作为缓存，属于可回收内存

## 四、内存与磁盘监控

```bash
# 内存使用情况
free -h                   # 人类可读格式
free -m                   # 以MB为单位

# 磁盘使用情况
df -h                     # 查看分区使用率和挂载点
df -i                     # 查看inode使用情况
du -sh /var/log/*         # 查看目录/文件大小
du -h --max-depth=1 /home # 一级子目录大小

# IO性能监控
iostat -x 1 5             # 每秒采集一次，共5次
iotop                     # 实时查看进程IO（需安装）

# iostat关键输出字段：
# %util  - 磁盘忙碌时间占比，接近100%说明磁盘瓶颈
# await  - IO请求平均等待时间(ms)
# svctm  - IO请求平均服务时间(ms)
# r/s w/s - 每秒读写请求数
```

## 五、网络配置与管理

```bash
# 查看网络接口
ip addr show              # 查看IP地址（推荐，替代ifconfig）
ip link show              # 查看网络接口状态
ss -tlnp                  # 查看监听端口（推荐，替代netstat）
ss -antp                  # 查看所有TCP连接

# 网络连通性测试
ping -c 4 8.8.8.8         # 发送4个ICMP包
traceroute google.com     # 路由追踪
mtr -r google.com         # 综合路由诊断

# DNS解析
nslookup example.com      # 查询DNS记录
dig example.com +short    # 详细DNS信息
cat /etc/resolv.conf      # 查看DNS配置

# 防火墙管理（firewalld）
firewall-cmd --list-all                          # 查看当前配置
firewall-cmd --add-port=8080/tcp --permanent     # 永久开放端口
firewall-cmd --reload                            # 重载配置

# 防火墙管理（iptables）
iptables -L -n -v                                # 列出规则
iptables -A INPUT -p tcp --dport 80 -j ACCEPT    # 允许80端口
```

## 六、Shell 基础

### 6.1 变量与环境变量

```bash
# 定义变量
NAME="production-server"
PORT=8080

# 环境变量
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
echo $JAVA_HOME
env | grep JAVA_HOME

# 命令替换
CURRENT_DATE=$(date +%Y%m%d)
FILE_COUNT=$(ls /var/log | wc -l)
```

### 6.2 输入输出重定向

```bash
# 标准输出重定向
ls -la > file_list.txt      # 覆盖写入
echo "log" >> app.log       # 追加写入

# 标准错误重定向
./script.sh 2> error.log
./script.sh > output.log 2>&1  # 输出和错误合并

# 管道
cat access.log | grep "ERROR" | wc -l
ps aux | grep nginx | awk '{print $2}'
```

### 6.3 文本处理三剑客

```bash
# grep - 文本搜索
grep -r "ERROR" /var/log/app/           # 递归搜索
grep -v "DEBUG" app.log | grep "ERROR"  # 排除DEBUG行
grep -c "404" access.log                # 统计匹配行数

# sed - 流编辑器
sed -i 's/http/https/g' nginx.conf     # 替换所有http为https
sed -n '10,20p' app.log                # 打印第10-20行
sed '/^$/d' config.txt                 # 删除空行

# awk - 文本分析
awk '{print $1}' access.log            # 打印第一列
awk '{sum+=$3} END {print sum}' data   # 求和
awk -F: '{print $1,$3}' /etc/passwd    # 指定分隔符
```

## 七、常用系统管理技巧

```bash
# 查看系统版本
cat /etc/os-release
uname -a

# 查看开机时间
uptime
who -b

# 查看历史命令
history | tail -20

# 定时任务
crontab -l                  # 查看当前用户的定时任务
crontab -e                  # 编辑定时任务
# 格式：分 时 日 月 周 命令
# 0 2 * * * /opt/backup.sh  # 每天凌晨2点执行备份

# 日志管理
journalctl -u nginx -f      # 实时查看nginx服务日志（systemd）
tail -f /var/log/syslog     # 系统日志
dmesg | tail -20            # 内核日志

# 软件包管理（CentOS/RHEL）
yum update                  # 更新所有软件包
yum install nginx           # 安装软件
yum list installed          # 查看已安装列表
rpm -qa | grep nginx        # 查询特定包

# 软件包管理（Ubuntu/Debian）
apt update && apt upgrade   # 更新源并升级
apt install nginx           # 安装软件
dpkg -l | grep nginx        # 查询已安装包
```

## 八、实践要点总结

1. **日常巡检**：定期检查负载（top/uptime）、磁盘使用率（df）、内存状态（free）和日志
2. **安全加固**：最小权限原则（chmod）、禁用root SSH登录、配置防火墙规则
3. **故障排查**：从网络连通性（ping/traceroute）、端口监听（ss）、进程状态（ps/top）、日志（tail/grep）四个维度逐层排查
4. **自动化思维**：重复操作编写脚本，通过crontab定时执行

掌握 Linux 系统管理是成为合格运维工程师的第一步，这些命令和工具构成了日常工作的基础工具包，需要在实际操作中不断熟练。
