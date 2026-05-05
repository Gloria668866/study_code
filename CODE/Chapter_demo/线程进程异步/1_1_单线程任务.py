import time
import os
import threading


# 正常函数（也称为同步函数）顺序执行
def download_file_sync(file_name):
    """
    一个同步下载任务。
    它会阻塞程序，直到任务完成。
    """
    pid = os.getpid()  # 获取当前进程ID
    thread_name = threading.current_thread().name  # 获取当前线程名
    print(f"进程ID: {pid}, 线程名: {thread_name} - 开始下载 {file_name}...")
    time.sleep(2)  # 模拟网络I/O耗时，这里是阻塞点
    print(f"进程ID: {pid}, 线程名: {thread_name} - 完成下载 {file_name}")


# 运行同步代码
if __name__ == '__main__':
    """同步执行所有下载任务"""
    start_time = time.time()
    files = ["文件A", "文件B", "文件C"]

    for file in files:
        download_file_sync(file)

    end_time = time.time()
    print(f"\n同步模式总耗时: {end_time - start_time:.2f} 秒")

"""
分析：任务串行执行，总耗时等于所有任务耗时之和（2s * 3 = 6s）。这是最简单但也最慢的模式。

"""