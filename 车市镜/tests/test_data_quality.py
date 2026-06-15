"""数据质量门禁单测（依赖 bi_demo.db）：GE 等价 expectations 套件须全通过。"""
import pytest

pytestmark = pytest.mark.integration   # 需 bi_demo.db，CI 默认跳过

from eval.data_quality import run       # noqa: E402


def test_all_expectations_pass():
    results = run()
    failed = [r["expectation"] for r in results if not r["success"]]
    assert not failed, f"数据质量未通过：{failed}"
