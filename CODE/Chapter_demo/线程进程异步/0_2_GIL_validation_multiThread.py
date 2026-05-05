import time
import os
import threading


def cpu_intensive_task_gil():
    """一个纯CPU密集型任务，用于验证GIL。"""
    pid = os.getpid() # 获取当前进程的进程ID
    thread_name = threading.current_thread().name # 获取当前线程的名称
    print(f"进程ID: {pid}, 线程: {thread_name} - 开始CPU计算任务...")

    count = 0
    # 纯粹的CPU计算，不涉及I/O
    for i in range(50000000):
        count += i

    print(f"进程ID: {pid}, 线程: {thread_name} - CPU计算任务完成。")


def main_gil_demo():
    print("--- 验证GIL: 多线程CPU密集型任务 ---")
    threads = []
    start_time = time.time()

    # 创建并启动两个线程来执行CPU密集型任务
    for i in range(2):
        t = threading.Thread(target=cpu_intensive_task_gil, name=f"Worker-{i}")
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    end_time = time.time()
    print(f"\n多线程总耗时: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    main_gil_demo()