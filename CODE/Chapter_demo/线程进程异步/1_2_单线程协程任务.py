import asyncio
import time
import threading
import os


# 模拟一个 I/O 密集型任务，比如下载文件
# 注意：这个函数没有使用 async/await，它是一个普通的同步函数，会阻塞。
def download_file(file_name, sleep_time):
    """
    模拟一个 I/O 密集型任务，比如下载文件
    """
    print(f"[{time.strftime('%H:%M:%S')}] 任务 '{file_name}' 开始下载...")
    time.sleep(sleep_time)  # 模拟网络等待
    print(f"[{time.strftime('%H:%M:%S')}] 任务 '{file_name}' 下载完成。")
    return f"文件 '{file_name}' 下载成功"


# 异步主程序，负责组织和调度任务
async def main():
    print("--- 模式: 单线程 + 协程（异步并发） ---")
    start_time = time.time()

    # 打印当前进程 ID 和线程 ID，证明所有协程都在同一个线程中运行
    print(f"主协程所在进程ID: {os.getpid()}")
    print(f"主协程所在线程ID: {threading.get_ident()}")
    print("-" * 20)

    # 1. 在一个线程中创建多个异步任务（协程，即任务清单）
    # 假设我们有一个真正的异步下载函数，它不会阻塞，所谓的不会阻塞就是指：它不会阻塞当前线程。
    # async def 开头的函数就是协程函数
    async def download_file_async(file_name, sleep_time):
        print(f"[{time.strftime('%H:%M:%S')}] 异步任务 '{file_name}' 开始下载...")
        print(f" -> 任务 '{file_name}' 所在线程ID: {threading.get_ident()}")

        # 遇到 await 关键字，线程会暂停，去处理其他任务，直到这里的 I/O 完成。
        # asyncio.sleep() 是一个异步操作，它不会阻塞线程，而是将控制权交出去。
        await asyncio.sleep(sleep_time)

        print(f"[{time.strftime('%H:%M:%S')}] 异步任务 '{file_name}' 下载完成。")
        return f"文件 '{file_name}' 下载成功"

    # 这三个任务都是协程
    # 每一个 asyncio.create_task() 都会创建一个协程任务
    task1 = asyncio.create_task(download_file_async('图片.jpg', 3))
    task2 = asyncio.create_task(download_file_async('视频.mp4', 5))
    task3 = asyncio.create_task(download_file_async('文档.pdf', 2))

    # 2. 等待所有任务完成
    # asyncio.gather 会并发地运行这些任务，等待它们全部完成
    results = await asyncio.gather(task1, task2, task3)

    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 20)
    print("\n--- 所有任务完成 ---")
    for res in results:
        print(f"结果: {res}")
    print(f"总耗时: {total_time:.2f} 秒")


# 运行主程序
if __name__ == "__main__":
    # asyncio.run() 是一个入口，它会创建一个事件循环来运行主协程
    asyncio.run(main())