import time
import os
import threading


def cpu_intensive_task_sync():
    """一个纯CPU密集型任务，模拟大量计算。"""
    pid = os.getpid() # 获取当前进程的ID
    thread_name = threading.current_thread().name # 获取当前线程的名称
    print(f"进程ID: {pid}, 线程: {thread_name} - 开始CPU计算任务...")

    task_start_time = time.time()
    count = 0
    # 纯粹的CPU计算，不涉及I/O
    for i in range(50000000):
        count += i
    task_end_time = time.time()

    print(f"进程ID: {pid}, 线程: {thread_name} - CPU计算任务完成。")
    return task_end_time - task_start_time


def main_single_thread():
    print("--- 模式: 单线程CPU密集型任务 ---")

    total_start_time = time.time()

    # 第一次任务
    task1_duration = cpu_intensive_task_sync()
    print(f"第一次任务耗时: {task1_duration:.2f} 秒\n")

    # 第二次任务
    task2_duration = cpu_intensive_task_sync()
    print(f"第二次任务耗时: {task2_duration:.2f} 秒\n")

    total_end_time = time.time()
    print(f"单线程总耗时（两次任务之和）: {total_end_time - total_start_time:.2f} 秒")


if __name__ == '__main__':
    main_single_thread()