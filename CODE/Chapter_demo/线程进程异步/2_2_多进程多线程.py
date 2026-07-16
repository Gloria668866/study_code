import time
import multiprocessing
import threading
import os
from concurrent.futures import ThreadPoolExecutor


# --- 1. 定义子进程要执行的函数 ---
# 1.1 模拟一个I/O密集型任务
def download_file(task_name, duration):
    """
    这是一个同步函数，模拟I/O等待，它会在每个线程中运行。
    """
    print(f"进程ID: {os.getpid()}, 线程名: {threading.current_thread().name} - 任务 '{task_name}' 开始下载...")
    time.sleep(duration)
    print(f"进程ID: {os.getpid()}, 线程名: {threading.current_thread().name} - 任务 '{task_name}' 下载完成。")
    return f"任务 '{task_name}' 下载成功"


# 1.2 这个函数将在独立的进程中运行，它负责管理线程池
def process_manager(tasks_list):
    """
    这个函数是子进程的入口点，它会在进程内部创建一个线程池，
    并将任务分发给线程来并发执行。
    """
    print(f"\n子进程ID: {os.getpid()}, 线程名: {threading.current_thread().name} - 启动...")

    # 使用 with 语句创建线程池
    # max_workers=3 表示最多创建3个线程来执行任务
    with ThreadPoolExecutor(max_workers=3) as executor:
        # executor.submit() 提交任务到线程池，它会立即返回一个 Future 对象
        futures = [executor.submit(download_file, task[0], task[1]) for task in tasks_list]

    # 获取所有任务结果
    results = [future.result() for future in futures]

    print(f"子进程ID: {os.getpid()} - 线程池任务结束。")
    return results


# --- 2. 主程序入口点 ---
if __name__ == '__main__':
    print("--- 模式: 多进程多线程 ---")
    print(f"主进程ID: {os.getpid()}") # 打印主进程ID
    print("--- 模式: 多进程多线程 ---")
    start_time = time.time()
    # 将所有任务分成两组，每组交给一个独立的进程处理
    tasks_group1 = [('A', 3), ('B', 5)]
    tasks_group2 = [('C', 2), ('D', 4)]

    # 2.1 创建进程池
    # multiprocessing.Pool(processes=2) 创建一个包含2个工作进程的进程池
    pool = multiprocessing.Pool(processes=2)

    # 2.2 将任务组提交给进程池
    # pool.map() 将 tasks_groups 的每个元素（一个任务列表）作为参数，
    # 分发给 pool 中的每个进程去执行 process_manager 函数
    tasks_groups = [tasks_group1, tasks_group2]
    all_results = pool.map(process_manager, tasks_groups)

    # 2.3 关闭并等待进程池
    # pool.close() 停止接受新的任务
    pool.close()
    # pool.join() 阻塞主程序，直到所有子进程都执行完毕
    pool.join()

    end_time = time.time()
    total_time = end_time - start_time

    print("\n--- 所有任务完成 ---")
    # 打印最终结果
    for group_results in all_results:
        for res in group_results:
            print(f"结果: {res}")
    print(f"总耗时: {total_time:.2f} 秒")