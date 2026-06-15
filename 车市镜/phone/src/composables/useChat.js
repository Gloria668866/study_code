import { reactive, ref, computed } from 'vue'
import { API, authHeaders } from '@/api/config.js'
import { postSSE } from '@/api/sse.js'

let _id = 0
const uid = () => `${Date.now()}-${++_id}`

function newAssistant() {
  return reactive({ id: uid(), role: 'assistant', status: 'thinking', intent: null, sql: '', columns: [], rows: [], chartPayload: null, insight: '', citations: [], error: null })
}

export function useChat() {
  const conversations = reactive([])
  const activeId = ref(null)
  const sending = ref(false)
  let handle = null

  const active = computed(() => conversations.find(c => c.id === activeId.value) || null)
  const messages = computed(() => active.value?.messages || [])

  function newConversation() {
    const conv = reactive({ id: uid(), title: '新对话', messages: [], serverConvId: null })
    conversations.unshift(conv)
    activeId.value = conv.id
  }

  function selectConversation(id) { activeId.value = id }

  async function deleteConversation(id) {
    const idx = conversations.findIndex(c => c.id === id)
    if (idx === -1) return
    const conv = conversations[idx]
    if (conv.serverConvId) {
      try { await fetch(API.historyDetail(conv.serverConvId), { method: 'DELETE', headers: authHeaders() }) } catch { return }
    }
    conversations.splice(idx, 1)
    if (activeId.value === id) activeId.value = conversations[0]?.id || null
  }

  function send(question) {
    const q = (question || '').trim()
    if (!q || sending.value) return
    let conv = active.value
    if (!conv) { newConversation(); conv = active.value }
    if (conv.messages.length === 0) conv.title = q.length > 20 ? q.slice(0, 20) + '…' : q

    conv.messages.push({ id: uid(), role: 'user', content: q })
    const msg = newAssistant()
    conv.messages.push(msg)
    sending.value = true

    const controller = new AbortController()
    handle = { abort: () => controller.abort() }

    postSSE(API.ask, { question: q }, {
      onEvent(ev) {
        switch (ev.type) {
          case 'intent': msg.intent = ev.intent; break
          case 'sql': msg.sql = ev.sql_text; break
          case 'rows': msg.columns = ev.columns || []; msg.rows = ev.rows || []; break
          case 'chart': msg.chartPayload = ev.chart || ev; break
          case 'insight': if (msg.status !== 'streaming') msg.status = 'streaming'; msg.insight += ev.delta || ''; break
          case 'citation': if (ev.citation) msg.citations.push(ev.citation); break
          case 'done': msg.status = 'done'; conv.serverConvId = ev.conversation_id; break
          case 'error': msg.status = 'error'; msg.error = ev; break
        }
      },
      onError(err) { msg.status = 'error'; msg.error = err; sending.value = false },
      onClose() { if (msg.status !== 'error') msg.status = 'done'; sending.value = false },
    }, controller.signal)
  }

  function stop() { handle?.abort(); sending.value = false }

  function loadHistory() {
    fetch(API.history, { headers: authHeaders() }).then(r => r.json()).then(d => {
      conversations.splice(0, conversations.length)
      for (const c of (d.conversations || [])) {
        conversations.push(reactive({ id: uid(), title: c.title, serverConvId: c.id, messages: [], loaded: false }))
      }
    }).catch(() => {})
  }

  return { conversations, activeId, messages, sending, newConversation, selectConversation, deleteConversation, send, stop, loadHistory }
}
