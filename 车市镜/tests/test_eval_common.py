"""评测工具单测：SQL 结果集等价比对（无序/列序无关/数值容差/有序）。"""
from eval.common import result_set_equal


def test_equal_ignores_colname_colorder_roworder():
    gold = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    pred = [{"x": 4, "y": 3}, {"x": 2, "y": 1}]   # 列名不同、列序行序都不同 → 值多重集相同
    assert result_set_equal(gold, pred)


def test_not_equal():
    assert not result_set_equal([{"a": 1}], [{"a": 2}])


def test_numeric_tolerance():
    assert result_set_equal([{"v": 1.0}], [{"v": "1.0000"}])


def test_ordered_flag():
    g = [{"a": 1}, {"a": 2}]
    p = [{"a": 2}, {"a": 1}]
    assert result_set_equal(g, p)                  # 默认无序 → 相等
    assert not result_set_equal(g, p, ordered=True)  # 有序 → 不等
