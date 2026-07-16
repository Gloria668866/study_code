# Shell脚本编程实战

## Shell基础

Shell是用户与Linux内核交互的命令行解释器。常见的Shell包括bash（Bourne Again Shell）、zsh、fish等。Linux系统默认使用bash。

## 变量与数据类型

### 变量定义和使用
```bash
#!/bin/bash
name="张三"
age=25
echo "姓名: $name, 年龄: $age"

# 只读变量
readonly PI=3.14159

# 删除变量
unset age
```

### 特殊变量

| 变量 | 含义 |
|------|------|
| $0 | 脚本名称 |
| $1～$9 | 位置参数 |
| $# | 参数个数 |
| $? | 上一条命令的退出状态（0成功） |
| $$ | 当前Shell进程ID |
| $@ | 所有参数列表 |

## 条件判断

### if语句
```bash
if [ $score -ge 90 ]; then
    echo "优秀"
elif [ $score -ge 60 ]; then
    echo "及格"
else
    echo "不及格"
fi
```

### 常用判断条件

| 运算符 | 含义 | 示例 |
|--------|------|------|
| -eq | 等于 | [ $a -eq $b ] |
| -ne | 不等于 | [ $a -ne $b ] |
| -gt | 大于 | [ $a -gt $b ] |
| -lt | 小于 | [ $a -lt $b ] |
| -z | 字符串为空 | [ -z "$str" ] |
| -n | 字符串非空 | [ -n "$str" ] |
| -f | 是文件 | [ -f "$file" ] |
| -d | 是目录 | [ -d "$dir" ] |

## 循环语句

```bash
# for循环
for i in {1..10}; do
    echo "第 $i 次循环"
done

# while循环
count=0
while [ $count -lt 5 ]; do
    echo "count: $count"
    ((count++))
done

# 遍历文件
for file in /var/log/*.log; do
    echo "处理文件: $file"
    gzip "$file"
done
```

## 函数

```bash
# 函数定义
check_service() {
    local service_name=$1
    if systemctl is-active --quiet $service_name; then
        echo "$service_name 运行中"
        return 0
    else
        echo "$service_name 未运行"
        return 1
    fi
}

# 函数调用
check_service "nginx"
check_service "mysql"
```

## 常用运维脚本示例

### 服务健康检查脚本
```bash
#!/bin/bash
SERVICES=("nginx" "mysql" "redis" "docker")

for svc in "${SERVICES[@]}"; do
    if systemctl is-active --quiet $svc; then
        echo "[OK] $svc"
    else
        echo "[FAIL] $svc - 尝试重启..."
        systemctl restart $svc
    fi
done
```

### 日志清理脚本
```bash
#!/bin/bash
LOG_DIR="/var/log/app"
RETENTION_DAYS=30

find "$LOG_DIR" -name "*.log" -mtime +$RETENTION_DAYS -exec gzip {} \;
find "$LOG_DIR" -name "*.gz" -mtime +90 -delete
echo "日志清理完成"
```

## Crontab定时任务

```bash
# 编辑定时任务
crontab -e

# 格式: 分 时 日 月 周 命令
0 2 * * * /scripts/backup.sh           # 每天凌晨2点备份
*/5 * * * * /scripts/health_check.sh   # 每5分钟健康检查
0 0 * * 0 /scripts/weekly_report.sh    # 每周日凌晨执行
```
