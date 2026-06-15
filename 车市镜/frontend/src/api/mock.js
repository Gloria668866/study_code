// 本地 Mock：后端未就绪时，按 PRD-2 §9.1 的事件节奏模拟 SSE 流，先把界面调通。
// 直接吐「规范事件」（与 events.js 归一化后的形态一致），组件层无感知。
// 注意：mock 数据是演示用占位（贴近真实量级但非实时），后端 live 后自动替换。

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// 把一段文字按字切片，模拟逐 token 流式
function tokenize(text) {
  const out = []
  for (const seg of text.split(/(\n)/)) {
    if (seg === '\n') { out.push('\n'); continue }
    for (let i = 0; i < seg.length; i += 2) out.push(seg.slice(i, i + 2))
  }
  return out
}

// —— 场景库 —— //
const SCENARIOS = [
  {
    match: (q) => /理想|小米|su7|谁卖|对比|vs/i.test(q),
    intent: 'sql',
    sql: `SELECT s.series_name, SUM(f.volume) AS total_volume
FROM fact_sales_rank f
JOIN dim_series s ON s.series_id = f.series_id
WHERE s.series_name LIKE '%理想%' OR s.series_name LIKE '%小米SU7%'
GROUP BY s.series_name
ORDER BY total_volume DESC
LIMIT 10;`,
    columns: ['车系', '累计销量'],
    rows: [['小米SU7', 460536], ['理想L6', 387948], ['理想L7', 227192], ['理想L9', 138204], ['理想MEGA', 41260]],
    chart: { default_type: 'bar', applicable_types: ['bar', 'hbar', 'line', 'pie'], dimension: '车系', measures: ['累计销量'], title: '车系销量对比' },
    insight: '小米SU7 以累计 46.05 万辆领跑，超过理想全系单车型最高的 L6（38.79 万辆）。\n\n归因：小米SU7 作为单一爆款车型集中放量，而理想以 L6/L7/L9 多车型分摊销量；若按品牌口径合计，理想全系约 79.5 万辆仍高于小米SU7。建议进一步看月度趋势判断后劲。',
  },
  {
    match: (q) => /top|前\s*\d+|排名|榜|纯电|插混|增程|销量/i.test(q),
    intent: 'sql',
    sql: `SELECT s.series_name, SUM(f.volume) AS total_volume
FROM fact_sales_rank f
JOIN dim_series s ON s.series_id = f.series_id
JOIN dim_date d ON d.date_id = f.date_id
WHERE f.new_energy_type = 1 AND d.year = 2025
GROUP BY s.series_name
ORDER BY total_volume DESC
LIMIT 10;`,
    columns: ['车系', '2025累计销量'],
    rows: [
      ['星愿', 465775], ['五菱宏光MINIEV', 435599], ['Model Y', 425337], ['海鸥', 388912],
      ['Model 3', 261480], ['元UP', 240117], ['海豚', 198640], ['小米SU7', 187233],
      ['AION S', 165902], ['零跑C10', 152018],
    ],
    chart: { default_type: 'bar', applicable_types: ['bar', 'hbar', 'line', 'pie'], dimension: '车系', measures: ['2025累计销量'], title: '2025 纯电销量 Top10' },
    insight: '2025 年纯电销量 Top10 中，星愿（46.58 万）、五菱宏光MINIEV（43.56 万）、Model Y（42.53 万）位列前三。\n\n归因：榜单呈「两端强」格局——低价代步（星愿、宏光MINIEV、海鸥）与中高端（Model Y/3、小米SU7）各占半壁，10-15 万主流家用纯电反而较少进入头部。建议关注小米SU7 作为新势力单车型已挤入 Top8 的势头。',
  },
]

const RAG_SCENARIO = {
  intent: 'rag',
  answer: '根据知识库中的行研报告，2025 年我国新能源乘用车渗透率预计达到 52%–55%，首次实现全年渗透率过半。\n\n报告给出的主要依据是：① 插混与增程车型对燃油车的加速替代；② 10–20 万元价格带纯电产品供给显著丰富；③ 多地以旧换新与购置税优惠延续。报告同时提示，渗透率增速将由高速扩张转入结构性分化阶段。',
  // 字段对齐后端 §5.5 引用结构：{doc_id, page_no, chunk_id, heading_path, title}
  citations: [
    { doc_id: 7, page_no: 12, chunk_id: 'c-7-12', title: '2025中国新能源汽车市场展望.pdf', heading_path: '二、需求侧 > 2.1 渗透率展望' },
    { doc_id: 7, page_no: 18, chunk_id: 'c-7-18', title: '2025中国新能源汽车市场展望.pdf', heading_path: '三、供给侧 > 3.2 价格带结构' },
    { doc_id: 11, page_no: 4, chunk_id: 'c-11-4', title: '乘联会月度零售数据点评.pdf', heading_path: '政策回顾 > 以旧换新与购置税' },
  ],
}

// 覆盖范围外的（合资/进口）品牌：模拟真实后端「查不到就老实说，绝不编造」
const OUT_OF_COVERAGE = ['奔驰', '宝马', '奥迪', '大众', '丰田', '本田', '日产', '马自达', '雷克萨斯', '沃尔沃', '凯迪拉克', '保时捷', '路虎', '捷豹', '别克', '雪佛兰', '福特', '现代', '起亚', '三菱', '英菲尼迪', '讴歌', '林肯']
const COVERED = ['比亚迪', '特斯拉', '理想', '蔚来', '小鹏', '零跑', '哪吒', '问界', '极氪', '小米', '吉利', '长安', '奇瑞', '长城', '五菱', '广汽', '埃安', '深蓝', '腾势', '方程豹', '仰望', '星愿', '海鸥', '海豚', 'model', '宏光', '智己', '阿维塔']

function pickScenario(question) {
  if (/报告|渗透率|政策|怎么看|文档|研报|为什么|解读/.test(question)) return { kind: 'rag' }
  // 覆盖范围外品牌（如「奔驰销量」）→ 走 nodata，与真实后端 B1 行为一致
  const q = question.toLowerCase()
  const out = OUT_OF_COVERAGE.find((b) => question.includes(b))
  const covered = COVERED.some((b) => q.includes(b.toLowerCase()))
  if (out && !covered) return { kind: 'nodata', brand: out }
  const s = SCENARIOS.find((sc) => sc.match(question))
  return s ? { kind: 'sql', s } : { kind: 'sql', s: SCENARIOS[1] } // 兜底给 Top10
}

/**
 * mock 版 SSE。签名与真实 transport 对齐：onEvent 收规范事件。
 */
export async function mockSSE(body, handlers, signal) {
  const { onEvent, onClose } = handlers
  const q = body.question || ''
  const aborted = () => signal?.aborted

  const picked = pickScenario(q)
  await sleep(420); if (aborted()) return

  if (picked.kind === 'nodata') {
    // 覆盖范围外品牌：出意图 + SQL（展示 Text2SQL 仍尝试了），但 0 行 → 不出图、不编造，老实告知
    onEvent({ type: 'intent', intent: 'sql', confidence: 0.9 })
    await sleep(600); if (aborted()) return
    onEvent({ type: 'sql', sql:
`SELECT s.series_name, SUM(f.volume) AS total_volume
FROM fact_sales_rank f
JOIN dim_series s ON s.series_id = f.series_id
JOIN dim_brand b ON b.brand_id = s.brand_id
WHERE b.brand_name LIKE '%${picked.brand}%'
GROUP BY s.series_id, s.series_name
ORDER BY total_volume DESC` })
    await sleep(520); if (aborted()) return
    const msg = `未查询到「${picked.brand}」的相关数据。「${picked.brand}」可能不在当前数据库覆盖范围内（目前覆盖 101 个品牌，以国产新能源为主）。\n\n可尝试：\n1. 换一个品牌或车系（如「比亚迪」「小米SU7」）\n2. 问更宽泛的问题（如「2025年纯电销量Top10」）`
    for (const tk of tokenize(msg)) { if (aborted()) return; onEvent({ type: 'insight', delta: tk }); await sleep(14) }
    onEvent({ type: 'done', msgId: Date.now() })
    onClose?.()
    return
  }

  if (picked.kind === 'rag') {
    onEvent({ type: 'intent', intent: 'rag', confidence: 0.91 })
    await sleep(700); if (aborted()) return
    // 知识脑：先流式答案，再给引用
    for (const tk of tokenize(RAG_SCENARIO.answer)) {
      if (aborted()) return
      onEvent({ type: 'insight', delta: tk })
      await sleep(22)
    }
    await sleep(300); if (aborted()) return
    // 与后端一致：逐条推 citation（一个事件一条引用）
    for (const c of RAG_SCENARIO.citations) {
      if (aborted()) return
      onEvent({ type: 'citation', citation: c })
      await sleep(60)
    }
    onEvent({ type: 'done', msgId: Date.now() })
    onClose?.()
    return
  }

  const s = picked.s
  onEvent({ type: 'intent', intent: 'sql', confidence: 0.94 })
  await sleep(650); if (aborted()) return
  onEvent({ type: 'sql', sql: s.sql })
  await sleep(520); if (aborted()) return
  onEvent({ type: 'rows', columns: s.columns, rows: s.rows })
  await sleep(360); if (aborted()) return
  onEvent({ type: 'chart', chart: s.chart })
  await sleep(420); if (aborted()) return
  for (const tk of tokenize(s.insight)) {
    if (aborted()) return
    onEvent({ type: 'insight', delta: tk })
    await sleep(20)
  }
  onEvent({ type: 'done', msgId: Date.now() })
  onClose?.()
}
