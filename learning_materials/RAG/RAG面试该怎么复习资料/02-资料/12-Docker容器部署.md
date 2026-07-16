# Docker容器部署

## 1 docker入门

### 1.Docker简介[¶](#1docker)

官方地址：https://docs.docker.com/

Docker 是一个基于go语言开发的开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包到一个可移植的容器中，然后发布到任何流行的 Linux 或者 Windows机器上，也可以实现虚拟化。

![虚拟机_Docker](file:///E:/baidu/%E5%8D%9A%E5%AD%A6%E8%B0%B7%E7%BD%91%E7%9B%98%E8%B5%84%E6%96%99/%E7%AC%AC%E4%B8%83%E9%98%B6%E6%AE%B5-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E9%A1%B9%E7%9B%AE1/%E6%89%A9%E5%B1%95%E8%AF%BE%E7%A8%8B/%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2/01-%E8%AE%B2%E4%B9%89/03-Docker%E5%AE%B9%E5%99%A8%E9%83%A8%E7%BD%B2/assets/%E8%99%9A%E6%8B%9F%E6%9C%BA_Docker-17149605206372.jpeg)

Docker 是一个用于开发、传送和运行应用程序的开放平台。Docker 使您能够将应用程序与基础设施分开，以便您可以快速交付软件。使用 Docker，您可以像管理应用程序一样管理基础设施。通过利用 Docker 的快速交付、测试和部署代码的方法，您可以显着减少编写代码和在生产中运行代码之间的延迟。Docker(opens new window)是个划时代的开源项目，它彻底释放了计算虚拟化的威力，极大提高了应用的维护效率，降低了云计算应用开发的成本！使用 Docker，可以让应用的部署、测试和分发都变得前所未有的高效和轻松！**无论是应用开发者、运维人员、还是其他信息技术从业人员，都有必要认识和掌握 Docker，节约有限的生命。**

![image-20240320112110769](file:///E:/baidu/%E5%8D%9A%E5%AD%A6%E8%B0%B7%E7%BD%91%E7%9B%98%E8%B5%84%E6%96%99/%E7%AC%AC%E4%B8%83%E9%98%B6%E6%AE%B5-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E9%A1%B9%E7%9B%AE1/%E6%89%A9%E5%B1%95%E8%AF%BE%E7%A8%8B/%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2/01-%E8%AE%B2%E4%B9%89/03-Docker%E5%AE%B9%E5%99%A8%E9%83%A8%E7%BD%B2/assets/image-20240320112110769.png)

### 2.Docker的主要特点[¶](#2docker)

1. **轻量化**：Docker 容器使用的资源非常少，相比虚拟机技术，一个完整的 Docker 镜像通常只有几十 MB，启动几乎是立即的。
2. **标准化**：Docker 对应用及其依赖进行标准化打包，解决了“在我机器上可以运行，在你机器上就不行”的问题。
3. **可移植性**：可以在不同的平台和不同的环境中运行，例如开发环境、测试环境和生产环境。
4. **版本管理和组件重用**：Docker 可以进行版本管理、组件重用、快速部署等。
5. **隔离性**：Docker 能够提供独立的运行环境，应用程序在 Docker 容器的运行和外部世界进行隔离。

### 3.为什么要使用Docker？[¶](#3docker)

1. **环境一致性**：在开发、测试和生产环境之间建立一致性，消除了“在我电脑上运行得好好的”这种情况。
2. **便于持续集成和持续部署**：Docker能够以最小的代价快速地启动和关闭，这让持续集成和持续部署变得非常简单。
3. **隔离性和安全性**：Docker容器之间彼此隔离，一个容器的崩溃不会影响到其他的容器，提供了额外的层次的安全性。
4. **微服务架构**：Docker非常适合微服务架构。每个微服务可以运行在自己的容器中，每个容器之间是相互隔离的，有自己独立的运行环境。
5. **资源利用率高**：Docker容器共享主机的内核，不需要像虚拟机那样为每个应用程序运行一个完整的操作系统，资源占用更少，启动更快。

因此，Docker 在软件开发、测试和运维中有着广泛的应用。

### 4.Docker核心概念[¶](#4docker)

Docker三大核心概念：镜像 Image、容器 Container、仓库 Repository

### （1）镜像（Image）[¶](#1image)

Docker 镜像（Image）可以被认为是 Docker 容器的模板。Docker 镜像是用于创建 Docker 容器的基础。简单来说，Docker 镜像就是一个只读的模板，用来创建 Docker 容器，一个镜像可以创建多个容器。

镜像包含了运行容器所需的所有内容，包括代码、运行时、库、环境变量和配置文件等。Docker 镜像是由文件系统叠加而成，每一层都代表 Dockerfile 中的一条指令，层与层之间是互相依赖的。

### （2）容器（Container）[¶](#2container)

Docker 容器（Container）是 Docker 镜像（Image）运行时的实体。容器可以被创建、启动、停止、删除、暂停等。简单来说，容器就是用镜像创建的运行实例。它可以被启动、开始、停止、移动和删除。每个容器之间是相互隔离的、保护的，每个容器都有自己的文件系统，每个容器之间运行的进程都是相互隔离的。

容器的定义和镜像几乎一样，也是一系列的层的集合，不同的是容器的最上面那一层是可读可写的，而镜像的最上层是只读的。

### （3）仓库（Repository）[¶](#3repository)

Docker 仓库（Repository）是用来保存镜像的地方。Docker 仓库可以被认为是代码控制中的代码仓库一样。Docker 用户可以在仓库中创建一个账户，存储和分享自己的镜像。也可以从 Docker 仓库中下载别人分享的镜像。

Docker 仓库分为公开和私有两种形式。公开的 Docker 仓库是 Docker 公司提供的，可以被所有用户使用。DockerHub 就是最知名的公开 Docker 仓库。私有的 Docker 仓库可以在本地部署，只能被内部用户使用。

总结来说，Docker 的工作流程是，首先从 Docker 仓库中获取 Docker 镜像，然后用这个镜像创建 Docker 容器，最后对容器进行操作（启动、停止等）。

### 5.Docker的核心架构[¶](#5docker)

在 Docker 主要有以下几部分组成：

1. Docker 客户端，负责与 Docker 守护进行通信；
2. Docker 守护进程，负责管理 Docker 镜像和 Docker 容器；
3. Docker 镜像，负责产生 Docker 容器实例；
4. Docker 容器，包含了应用程序和其所依赖环境。
5. Docker Hub，已经制作好了很多镜像

![image-20240320111926185](file:///E:/baidu/%E5%8D%9A%E5%AD%A6%E8%B0%B7%E7%BD%91%E7%9B%98%E8%B5%84%E6%96%99/%E7%AC%AC%E4%B8%83%E9%98%B6%E6%AE%B5-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E9%A1%B9%E7%9B%AE1/%E6%89%A9%E5%B1%95%E8%AF%BE%E7%A8%8B/%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2/01-%E8%AE%B2%E4%B9%89/03-Docker%E5%AE%B9%E5%99%A8%E9%83%A8%E7%BD%B2/assets/image-20240320111926185.png)

链接：https://hub.docker.com/

## 2 Centos Docker安装

192.168.88.3这个服务器

### 1 Docker自动化安装

**官方的一键安装方式：**

```sh
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
```

**国内daocloud一键安装命令：**

```sh
curl -sSL https://get.daocloud.io/docker | sh
```

执行上述任一条命令，耐心等待即可完成Docker的安装。

往往会因为网络的问题而报错



### 2 Docker手动安装

[docker 安装及部署常用指令（基础篇）_+ sh -c 'yum install -y -q docker-ce docker-ce-cli-CSDN博客](https://blog.csdn.net/jaja_zz/article/details/112676828)

```shell
# 安装依赖工具 
# yum-utils提供yum-config-manager
# device mapper存储驱动程序需要device-mapper-persistent-data和lvm2。
sudo yum install -y yum-utils device-mapper-persistent-data lvm2

# 配置镜像源， 我们使用的是阿里云的源
yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
yum makecache fast

# 安装 Docker
yum install docker-ce docker-ce-cli containerd.io
```

报错：Server: Cannot connect to the Docker daemon at unix:///run/podman/podman.sock. Is the docker daemon running?

解决：

```shell
# 临时设置别名
alias docker=podman

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo "alias docker=podman" >> ~/.bashrc
source ~/.bashrc

# 现在可以使用 docker 命令（实际调用的是 podman）
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.3.4
```

安装命令执行成功之后，使用下面命令来管理Docker 守护进程：

```shell
# 查看 docker 版本
docker version
# 启动 docker 服务
systemctl start docker 
# 关闭 docker 服务
systemctl stop docker
# 重启 docker 服务
systemctl restart docker
# 查看 docker 状态
systemctl status docker
# 设置开机启动
systemctl enable docker
```

## 3 Windows Docker安装

[windows 环境下安装 Milvus_windows安装milvus-CSDN博客](https://blog.csdn.net/xu_hui123/article/details/146583355)

[Milvus 安装和启动指南_milvus安装-CSDN博客](https://blog.csdn.net/weixin_67868534/article/details/150534751)

[一文带你入门向量数据库milvus：含docker安装、milvus安装使用、attu 可视化，完整指南启动 Milvus 进行了向量相似度搜索-腾讯云开发者社区-腾讯云](https://cloud.tencent.com.cn/developer/article/2338320)

```
如果docker compose up -d 拉取速度太慢，更换镜像源引擎

{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker-0.unsee.tech",
    "https://docker.m.daocloud.io"
  ],
  "features": { "buildkit": true }
}
```

![image-20260201154228704](E:\01-线下实训资料\02-资料\assets\image-20260201154228704.png)

## 4 Docker镜像操作

```sh
# 1. 查看镜像
docker images             # 查看所有镜像
docker images -q          # 只查看镜像的ID
docker images --no-trunc  # 显示镜像完整信息

# 2. 搜索镜像
docker search 镜像名

# 3. 下载镜像
docker pull 镜像名:版本     # 不指定 TAG, 则默认使用 latest

# 4. 运行镜像
docker run -it 镜像名:版本 程序                # 交互式运行容器
docker run -it --name=标签名 镜像名:版本 程序   # 容器指定名字
docker run -itd 镜像名:版本 程序               # 后台运行容器

# 5. 删除镜像
docker rmi -f 镜像名                   # 删除指定镜像
docker rmi -f 镜像ID                   # 删除指定镜像
docker rmi -f $(docker images -qa)    # 删除所有镜像

# 6. 保存镜像
docker save 镜像名:版本 -o xxx.tar

"""
[root@bogon ~]# docker save alpine:latest -o myalpine.tar
[root@bogon ~]# ls
anaconda-ks.cfg  myalpine.tar
"""

# 7. 加载镜像
docker load -i xxx.tar

"""
[root@bogon ~]# docker load -i myalpine.tar
24302eb7d908: Loading layer [==================================================>]  5.811MB/5.811MB

REPOSITORY   TAG       IMAGE ID       CREATED        SIZE
alpine       latest    e66264b98777   3 weeks ago    5.53MB
centos       latest    5d0da3dc9764   9 months ago   231MB
"""
```



## 5 Docker快速入手

在这一节，我们通过一个小实验来快速入门 Docker 的使用方法。大致的步骤如下:

1. 查看 Docker 服务运行状态;
2. 查看本地镜像;
3. 从 Docker Hub 拉取基础镜像, 我们此处选择 ubuntu:18.04 镜像;
4. 再次查看本地镜像;
5. 使用 ubuntu:18.04 镜像构建容器，并交互式运行容器；
6. 在容器内部执行 LS 命令;
7. 退出容器;
8. 查看本地容器实例;
9. 再次启动停止的容器;
10. 退出并停止容器.

执行命令如下:

```sh
# 0. 查看 Docker 服务运行状态;
systemctl status docker

# 1. 查看本地镜像;
docker images
"""
REPOSITORY   TAG       IMAGE ID   CREATED   SIZE
"""

# 2. 从 Docker Hub 拉取基础镜像, 我们此处选择 ubuntu 镜像;
docker search ubuntu
docker search ubuntu --no-trunc
docker pull ubuntu
"""
Using default tag: latest
latest: Pulling from library/ubuntu
405f018f9d1d: Pull complete
Digest: sha256:b6b83d3c331794420340093eb706a6f152d9c1fa51b262d9bf34594887c2c7ac
Status: Downloaded newer image for ubuntu:latest
docker.io/library/ubuntu:latest
"""

# 3. 再次查看本地镜像;
docker images
docker image ls
"""
REPOSITORY   TAG       IMAGE ID       CREATED      SIZE
ubuntu       latest    27941809078c   9 days ago   77.8MB
"""

# 4. 使用 ubuntu 镜像构建容器，并交互式运行容器，并在容器中执行 LS 命令；
docker run -it ubuntu:latest /bin/bash
"""
root@abcced6d5ee8:/# ls
bin  boot  dev  etc  home  lib  lib32  lib64  libx32  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
"""

# 5. 退出容器;
exit
"""
exit
[root@bogon docker]#
"""

# 6. 查看本地容器实例;
docker ps
docker ps -a
"""
CONTAINER ID   IMAGE           COMMAND       CREATED              STATUS                      PORTS     NAMES
abcced6d5ee8   ubuntu:latest   "/bin/bash"   About a minute ago   Exited (0) 35 seconds ago             sad_montalcini
"""

# 7. 再次启动停止的容器;
docker start 容器ID
"""
[root@bogon docker]# docker start abcced6d5ee8
abcced6d5ee8
"""

# 8. 再次进入容器
docker exec -it abcced6d5ee8 /bin/bash
"""
root@abcced6d5ee8:/# ls
bin  boot  dev  etc  home  lib  lib32  lib64  libx32  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
"""

# 9. 退出容器, 并停止容器
exit
docker stop 容器ID
"""
[root@bogon docker]# docker ps
CONTAINER ID   IMAGE           COMMAND       CREATED         STATUS         PORTS     NAMES
abcced6d5ee8   ubuntu:latest   "/bin/bash"   9 minutes ago   Up 6 minutes             sad_montalcini
[root@bogon docker]# docker stop abcced6d5ee8
abcced6d5ee8
[root@bogon docker]# docker exec -it abcced6d5ee8 /bin/bash
Error response from daemon: Container abcced6d5ee8e6980f8271d3e78d18d0dff708e377c22c7798006a133ac73559 is not running
"""
```

常见报错1：pymilvus.exceptions.MilvusException: <MilvusException: (code=2, message=Fail connecting to server on localhost:19530, illegal connection params or server unavailable)>

![image-20250928191002518](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250928191002518.png)

## 6 Docker容器操作

容器操作主要包括:

1. 查看容器
2. 启动容器
3. 停止容器
4. 删除容器
5. 进入容器
6. 容器导出
7. 容器导入

```sh
# 1. 查看容器
docker ps       # 查看正在运行的容器实例
docker ps -a    # 查看正在运行或者已停止的容器实例

# 2. 运行容器
docker start 容器ID     # 启动容器
docker restart 容器ID   # 重启容器

# 3. 停止容器
docker stop 容器ID

# 4. 删除容器
docker rm -f 容器ID              # 删除指定容器
docker rm -f $(docker ps -qa)   # 删除所有容器

# 5. 进入容器
# attach 退出终端会导致容器停止
# exec 不会导致容器停止
docker attach 容器ID
docker exec -it 容器ID /bin/bash

# 6. 容器导出
docker export 容器ID > xxx.tar

# 7. 容器导入
docker import xxx.tar xxx:tag
```

## 7 Docker自动构建镜像（了解）

### 1. Dockfile 编写[¶](#1-dockfile)

我们也可以使用 Dockerfile 构建镜像，Dockerfile 其实就是把我们前面的一系列安装、配置命令写到一个文件中，通过 docker build 命令，一键完成镜像的构建。接下来，我们以 python3.7.5 作为基础镜像，来构建我们自己的镜像。

```sh
# 继承的基础镜像
FROM python:3.7.5
MAINTAINER "wechat:chinesecpp, email:chinacpp@hotmail.com"

# 安装 app 需要的 Python 包
RUN pip install pandas flask scikit-learn jieba zhconv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置工作目录
WORKDIR /root/app

# COPY 命令使用的是相对路径
COPY app/ /root/app

# 显式声明容器服务监听的端口
EXPOSE 5000

# 当启动容器时默认执行的命令
CMD ["python", "app.py"]
```

接下来，使用下面命令构建 Docker 镜像：

```sh
docker build -t spam:1.0 .
```

### 2. 镜像使用[¶](#2)

镜像构建完成之后，启动镜像创建容器实例：

```sh
docker run -d -p 8000:5000 spam:1.0
```

持久化本地存储镜像:

```sh
docker save spam:1.0 -o spam.tar
```

## 8 启动Milvus服务

报错怎么办？

![image-20251020170841037](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20251020170841037.png)

解决方案：

```cmd
# 1 查看docker的版本
PS D:\01-software\milvus> docker-compose --version
Docker Compose version v2.39.2-desktop.1
PS D:\01-software\milvus>
```

2 把梯子打开；

docker设置引擎如下；

![image-20251020174707017](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20251020174707017.png)

**启动Milvus相关服务；**

这个命令会启动 Milvus 和它的依赖服务，如 Zookeeper 和 etcd。

![image-20251020174851809](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20251020174851809.png)

![image-20251020175336639](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20251020175336639.png)

```
PS D:\01-software\milvus> docker-compose -f docker-compose.yml up -d
time="2025-10-20T17:45:01+08:00" level=warning msg="D:\\01-software\\milvus\\docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion"
[+] Running 15/15
 ✔ standalone Pulled                                                                                             128.4s
   ✔ d0c0ceb106a0 Pull complete                                                                                    0.4s
   ✔ d7388a78c1a2 Pull complete                                                                                   29.5s
   ✔ c982406aa267 Pull complete                                                                                   29.4s
   ✔ 7dfb73baa5d3 Pull complete                                                                                    1.5s
   ✔ 232928bf0cec Pull complete                                                                                   20.9s
   ✔ ab1c981276db Pull complete                                                                                  122.4s
   ✔ 7646c8da3324 Pull complete                                                                                   20.2s
 ✔ minio Pulled                                                                                                   19.4s
   ✔ c77f9df7200e Pull complete                                                                                    0.9s
   ✔ b40a74db8d08 Pull complete                                                                                    1.6s
   ✔ 14d4e804ec54 Pull complete                                                                                    1.5s
   ✔ 3bd551264400 Pull complete                                                                                   13.5s
   ✔ 188c0c94c7c5 Pull complete                                                                                    5.4s
   ✔ ed66f2d577c3 Pull complete                                                                                    5.5s
[+] Running 4/4
 ✔ Network milvus               Created                                                                            0.2s
 ✔ Container milvus-minio       Started                                                                            1.0s
 ✔ Container milvus-etcd        Started                                                                            1.0s
 ✔ Container milvus-standalone  Started                                                                            1.4s
PS D:\01-software\milvus>
```

![image-20250929152622510](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929152622510.png)
