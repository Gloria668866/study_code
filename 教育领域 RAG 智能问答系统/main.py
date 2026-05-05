# -*- coding:utf-8 -*-
"""
EduRAG 集成问答系统 — 主入口。
整合三条检索路径，按优先级降级:
  1. MySQL BM25 精确匹配  →  直接返回
  2. RAG 向量检索 + DeepSeek V4 生成  →  流式返回
  3. 兜底  →  提示未找到
用法:
  python main.py                          交互式 REPL
  python main.py -q "什么是AI"            单次查询
  python main.py -q "..." -s ai           按学科过滤
  python main.py -q "..." --session-id ID 携带会话历史
  python main.py --history SESSION_ID     查看历史记录
"""
import argparse
import sys
import os
import uuid
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from base import logger
from qa_service import IntegratedQASystem
# CLI

def _print_header():
    print("=" * 56)
    print("  EduRAG 集成问答系统")
    print("  检索路径: MySQL BM25 → RAG → DeepSeek V4")
    print("=" * 56)


def _print_history(system: IntegratedQASystem, session_id: str):
    history = system.get_session_history(session_id)
    if not history:
        print(f"  会话 {session_id} 无历史记录")
        return
    print(f"\n  会话 {session_id} 最近 {len(history)} 轮对话:")
    for i, h in enumerate(history, 1):
        print(f"    [{i}] Q: {h['question']}")
        print(f"        A: {h['answer'][:120]}{'...' if len(h['answer']) > 120 else ''}")


def _repl(system: IntegratedQASystem, session_id: str = None):
    """交互式 REPL。"""
    valid_sources = system.config.VALID_SOURCES
    _print_header()
    print(f"  支持学科: {', '.join(valid_sources)}")
    print("  输入 'exit' 退出 | 'history' 查看历史 | 'new' 新会话")
    print()

    if session_id is None:
        session_id = str(uuid.uuid4())
    print(f"  会话 ID: {session_id}")

    while True:
        try:
            raw = input("\n  请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not raw:
            continue
        if raw.lower() == "exit":
            print("  再见！")
            break
        if raw.lower() == "history":
            _print_history(system, session_id)
            continue
        if raw.lower() == "new":
            session_id = str(uuid.uuid4())
            print(f"  新会话 ID: {session_id}")
            continue

        # 解析 source 前缀，如 "ai: 什么是深度学习"
        source_filter = None
        query = raw
        if ":" in raw:
            prefix, rest = raw.split(":", 1)
            prefix = prefix.strip()
            if prefix in valid_sources:
                source_filter = prefix
                query = rest.strip()
                print(f"  [学科过滤: {source_filter}]")

        print(f"\n  回答: ", end="", flush=True)
        try:
            for token in system.query(query, source_filter=source_filter, session_id=session_id):
                print(token, end="", flush=True)
            print()
        except Exception as e:
            logger.error(f"查询失败: {e}")
            print(f"\n  处理出错，请重试。")

    return session_id


def main():
    parser = argparse.ArgumentParser(
        description="EduRAG 集成问答系统 — MySQL BM25 + RAG + DeepSeek V4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                           交互式 REPL
  python main.py -q "什么是AI"            单次查询
  python main.py -q "什么是AI" -s ai       按学科过滤
  python main.py -q "什么是AI" --session-id 550e8400  携带会话历史
  python main.py --history 550e8400        查看会话历史
        """,
    )
    parser.add_argument("-q", "--query", type=str, help="单次查询内容")
    parser.add_argument("-s", "--source", type=str, help="学科过滤 (ai/java/test/ops/bigdata)")
    parser.add_argument("--session-id", type=str, help="会话 ID，用于多轮对话")
    parser.add_argument("--history", type=str, metavar="SESSION_ID", help="查看指定会话的历史记录")

    args = parser.parse_args()

    try:
        system = IntegratedQASystem()
    except Exception as e:
        print(f"系统初始化失败: {e}")
        sys.exit(1)

    try:
        # --history 模式
        if args.history:
            _print_header()
            _print_history(system, args.history)
            return

        # 单次查询模式
        if args.query:
            session_id = args.session_id or str(uuid.uuid4())
            print(f"问题: {args.query}")
            if args.source:
                print(f"学科: {args.source}")
            print(f"回答: ", end="", flush=True)
            for token in system.query(args.query, source_filter=args.source, session_id=session_id):
                print(token, end="", flush=True)
            print()
            return

        # 交互式 REPL
        session_id = args.session_id
        _repl(system, session_id)

    finally:
        system.close()


if __name__ == "__main__":
    main()
