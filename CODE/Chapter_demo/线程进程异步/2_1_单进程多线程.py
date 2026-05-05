import time
import threading
from concurrent.futures import ThreadPoolExecutor
import os


# 1.1 模拟一个同步的 I/O 密集型任务
def io_bound_task(task_name, duration):
    """
    这是一个同步函数，模拟 I/O 等待。
    """
    # 打印当前任务所在的进程ID和线程名
    print(f"进程ID: {os.getpid()}, 线程名: {threading.current_thread().name} - 任务 '{task_name}' 开始下载...")

    # time.sleep() 是一个阻塞操作，会阻塞当前线程
    time.sleep(duration)

    print(f"进程ID: {os.getpid()}, 线程名: {threading.current_thread().name} - 任务 '{task_name}' 下载完成。")
    return f"任务 '{task_name}' 下载成功"


# 1.2 主程序：创建并管理多个线程
if __name__ == '__main__':
    print("--- 模式: 单进程多线程 ---")
    start_time = time.time()

    # 获取主进程ID，可以看到所有任务都在同一个进程中
    main_pid = os.getpid()
    print(f"主进程ID: {main_pid}")

    # 将所有任务放在一个列表中
    tasks = [('A', 3), ('B', 5), ('C', 2), ('D', 4)]

    # 1.2.1 创建线程池
    # ThreadPoolExecutor 用于在单个进程内创建并管理线程。
    # max_workers=4 表示最多创建4个线程来执行任务。
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 1.2.2 将任务提交到线程池
        # executor.submit() 提交任务到线程池，并返回一个 Future 对象  # task[0]为任务名称, task[1]取值为任务的持续时间
        futures = [executor.submit(io_bound_task, task[0], task[1]) for task in tasks]

    # 1.2.3 获取所有任务的结果
    # future.result() 会阻塞，直到对应的任务完成
    results = [future.result() for future in futures]

    end_time = time.time()
    total_time = end_time - start_time

    print("\n--- 所有任务完成 ---")
    for res in results:
        print(f"结果: {res}")
    print(f"总耗时: {total_time:.2f} 秒")