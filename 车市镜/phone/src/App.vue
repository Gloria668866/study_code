<script setup>
import { ref, computed, nextTick, watch, onMounted } from 'vue'
import { useAuth, isAuthed } from './composables/useAuth.js'
import { useChat } from './composables/useChat.js'

const { state: auth, login, register, logout } = useAuth()
const { conversations, activeId, messages, sending, newConversation, selectConversation, deleteConversation, send, stop, loadHistory } = useChat()

const tab = ref('home')
const showKb = ref(false)

const TABS = [
  { key: 'home', label: '首页', icon: '🏠' },
  { key: 'explore', label: '探索', icon: '🔍' },
  { key: 'chat', label: '对话', icon: '💬' },
  { key: 'kb', label: '知识库', icon: '📚' },
]

const lastQuestion = computed(() => {
  for (let i = messages.value.length - 1; i >= 0; i--) if (messages.value[i].role === 'user') return messages.value[i].content
  return ''
})
function onAsk(q) { if (!activeId.value) newConversation(); send(q); tab.value = 'chat' }
function onNew() { newConversation(); tab.value = 'chat' }

onMounted(() => { if (isAuthed.value) loadHistory() })

// === 登录表单 ===
const loginMode = ref('login')
const loginForm = ref({ username: '', password: '', nickname: '' })
const busy = ref(false)
async function doLogin() { busy.value = true; await login(loginForm.value.username, loginForm.value.password); busy.value = false }
async function doRegister() { busy.value = true; await register(loginForm.value.username, loginForm.value.password, loginForm.value.nickname); busy.value = false }

// === 对话输入 ===
const inputText = ref(''); const inputEl = ref(null)
function resize() { const el = inputEl.value; if (!el) return; el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 100) + 'px' }
watch(inputText, () => nextTick(resize))
function chatSend() { const q = inputText.value.trim(); if (!q || sending.value) return; onAsk(q); inputText.value = ''; nextTick(resize) }

// === 搜索 ===
const searchKw = ref('')
</script>

<template>
  <!-- ===== 登录页 ===== -->
  <div v-if="!isAuthed" class="login-page">
    <div class="login-card">
      <h1 class="login-brand">车<span class="rd">市</span>镜</h1>
      <p class="login-sub">新能源车市情报终端</p>
      <div class="login-tabs">
        <button :class="{ on: loginMode === 'login' }" @click="loginMode = 'login'">登录</button>
        <button :class="{ on: loginMode === 'register' }" @click="loginMode = 'register'">注册</button>
      </div>
      <div v-if="auth.error" class="login-err">{{ auth.error }}</div>
      <input v-model="loginForm.username" class="login-input" placeholder="用户名" />
      <input v-model="loginForm.password" class="login-input" type="password" placeholder="密码" />
      <input v-if="loginMode === 'register'" v-model="loginForm.nickname" class="login-input" placeholder="昵称（选填）" />
      <button class="login-btn" :disabled="busy" @click="loginMode === 'login' ? doLogin() : doRegister()">
        {{ busy ? '…' : loginMode === 'login' ? '登录' : '注册' }}
      </button>
    </div>
  </div>

  <!-- ===== 主界面 ===== -->
  <div v-else class="app">
    <!-- 页面内容 -->
    <main class="main">
      <!-- 首页 -->
      <div v-if="tab === 'home'" class="page home-page">
        <div class="home-header">
          <h1>车市镜</h1>
          <span class="home-sub">新能源车市情报终端</span>
        </div>
        <div class="search-bar" @click="tab = 'chat'">
          <span>🔍</span><span style="color:#9ca3af">搜索品牌、车系或提问…</span>
        </div>
        <div class="kpi-row">
          <div class="kpi-item"><b>409</b><span>车系</span></div>
          <div class="kpi-item"><b>101</b><span>品牌</span></div>
          <div class="kpi-item"><b>8,072</b><span>数据</span></div>
        </div>
        <div class="section-title">热门查询</div>
        <div class="hot-grid">
          <button v-for="h in ['2025年纯电销量Top10','比亚迪各车系今年销量','小米SU7和Model 3谁卖得多','新能源月度销量趋势','增程车型市场占比','2025年新上榜车型']" :key="h" class="hot-card" @click="onAsk(h)">{{ h }}</button>
        </div>
      </div>

      <!-- 探索页 -->
      <div v-if="tab === 'explore'" class="page">
        <div class="page-title">探索品牌与车系</div>
        <input v-model="searchKw" class="search-input" placeholder="搜索品牌或车系…" />
        <div class="brand-grid">
          <button v-for="b in ['比亚迪','特斯拉','理想','蔚来','小鹏','零跑','哪吒','问界','极氪','小米汽车','吉利','长安','奇瑞','长城','五菱']" :key="b" class="brand-card" @click="onAsk(b+'各车系今年销量')">
            <span class="brand-icon">🚗</span>
            <span class="brand-name">{{ b }}</span>
          </button>
        </div>
      </div>

      <!-- 对话页 -->
      <div v-if="tab === 'chat'" class="page chat-page">
        <div class="chat-top">
          <span class="chat-title">对话</span>
          <button class="chat-new" @click="onNew()">＋ 新建</button>
        </div>
        <div class="chat-body">
          <div v-if="!messages.length" class="chat-empty">💬 输入问题开始对话</div>
          <div v-for="m in messages" :key="m.id" :class="['msg', m.role]">
            <div v-if="m.role === 'user'" class="msg-user">{{ m.content }}</div>
            <div v-else class="msg-assistant">
              <div v-if="m.status === 'thinking'" class="msg-loading">分析中…</div>
              <div v-if="m.rows.length" class="msg-chart-hint">📊 {{ m.rows.length }} 条数据</div>
              <div v-if="m.insight" class="msg-text">{{ m.insight }}</div>
              <div v-if="m.status === 'error'" class="msg-err">{{ m.error?.message || '出错了' }}</div>
            </div>
          </div>
        </div>
        <div class="chat-input-bar">
          <textarea ref="inputEl" v-model="inputText" rows="1" placeholder="输入问题…" @keydown.enter.exact.prevent="chatSend" />
          <button v-if="!sending" class="chat-send" :disabled="!inputText.trim()" @click="chatSend">发送</button>
          <button v-else class="chat-send chat-stop" @click="stop()">■</button>
        </div>
      </div>

      <!-- 知识库页 -->
      <div v-if="tab === 'kb'" class="page">
        <div class="page-title">知识库</div>
        <p class="muted" style="padding:16px;text-align:center">上传研报、政策文档、行业报告，<br/>即可就文档内容提问。</p>
        <button class="kb-upload-btn" @click="showKb = true">📄 上传文档</button>
      </div>
    </main>

    <!-- 底部导航 -->
    <nav class="tabs">
      <button v-for="t in TABS" :key="t.key" :class="{ active: tab === t.key }" @click="tab = t.key">
        <span class="tab-icon">{{ t.icon }}</span>
        <span class="tab-label">{{ t.label }}</span>
      </button>
    </nav>
  </div>
</template>

<style scoped>
/* === 全局布局 === */
.app { height: 100%; display: flex; flex-direction: column; background: #f8f8f7; }
.main { flex: 1; overflow-y: auto; -webkit-overflow-scrolling: touch; }
.page { min-height: 100%; padding: 16px; animation: fadeUp .2s ease both; }

/* === 底部导航 === */
.tabs { flex: none; display: flex; background: #fff; border-top: 1px solid #e5e7eb; padding: 4px 0 env(safe-area-inset-bottom); }
.tabs button { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 2px; padding: 6px 4px; font-size: 10px; color: #9ca3af; }
.tabs button.active { color: #dc2626; }
.tab-icon { font-size: 20px; }
.tab-label { font-size: 10px; font-weight: 600; }

/* === 登录页 === */
.login-page { height: 100%; display: flex; align-items: center; justify-content: center; background: #f8f8f7; padding: 24px; }
.login-card { width: 100%; max-width: 360px; }
.login-brand { text-align: center; font-size: 32px; font-weight: 800; margin: 0 0 4px; }
.login-brand .rd { color: #dc2626; }
.login-sub { text-align: center; font-size: 11px; color: #dc2626; letter-spacing: .1em; margin: 0 0 24px; font-weight: 600; }
.login-tabs { display: flex; gap: 4px; margin-bottom: 16px; }
.login-tabs button { flex: 1; padding: 10px; font-size: 14px; font-weight: 600; color: #9ca3af; background: #e5e7eb; border-radius: 8px; }
.login-tabs button.on { background: #dc2626; color: #fff; }
.login-err { font-size: 12px; color: #dc2626; background: #fef2f2; padding: 8px 12px; margin-bottom: 12px; border-radius: 6px; }
.login-input { width: 100%; padding: 12px 14px; font-size: 14px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 8px; outline: none; }
.login-input:focus { border-color: #dc2626; }
.login-btn { width: 100%; padding: 14px; font-size: 15px; font-weight: 700; background: #dc2626; color: #fff; border-radius: 10px; margin-top: 8px; }
.login-btn:disabled { opacity: .5; }

/* === 首页 === */
.home-header { margin-bottom: 16px; }
.home-header h1 { font-size: 24px; font-weight: 800; margin: 0; }
.home-sub { font-size: 11px; color: #dc2626; font-weight: 600; letter-spacing: .06em; }
.search-bar { display: flex; align-items: center; gap: 10px; padding: 14px 16px; background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; margin-bottom: 16px; font-size: 14px; }
.kpi-row { display: flex; background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; margin-bottom: 24px; }
.kpi-item { flex: 1; text-align: center; padding: 14px 8px; }
.kpi-item + .kpi-item { border-left: 1px solid #f3f4f6; }
.kpi-item b { display: block; font-size: 22px; font-weight: 800; font-variant-numeric: tabular-nums; }
.kpi-item span { display: block; font-size: 10px; color: #9ca3af; margin-top: 2px; }
.section-title { font-size: 12px; font-weight: 700; color: #6b7280; margin-bottom: 10px; letter-spacing: .06em; }
.hot-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.hot-card { padding: 10px 14px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; font-size: 13px; font-weight: 600; }
.hot-card:active { background: #fef2f2; border-color: #fca5a5; }

/* === 探索页 === */
.page-title { font-size: 18px; font-weight: 800; margin-bottom: 12px; }
.search-input { width: 100%; padding: 12px 14px; font-size: 14px; background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; margin-bottom: 16px; outline: none; }
.search-input:focus { border-color: #dc2626; }
.brand-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.brand-card { display: flex; align-items: center; gap: 6px; padding: 10px 14px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; font-size: 13px; font-weight: 600; }
.brand-card:active { background: #fef2f2; }
.brand-icon { font-size: 18px; }
.brand-name { color: #111827; }

/* === 对话页 === */
.chat-page { display: flex; flex-direction: column; padding: 0; height: 100%; }
.chat-top { flex: none; display: flex; align-items: center; padding: 10px 16px; background: #fff; border-bottom: 1px solid #e5e7eb; }
.chat-title { flex: 1; font-size: 16px; font-weight: 700; }
.chat-new { color: #dc2626; font-size: 14px; font-weight: 600; }
.chat-body { flex: 1; overflow-y: auto; padding: 12px; }
.chat-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: #9ca3af; font-size: 14px; }
.msg { margin-bottom: 16px; }
.msg-user { display: inline-block; max-width: 80%; padding: 10px 14px; background: #dc2626; color: #fff; border-radius: 16px 16px 4px 16px; font-size: 14px; float: right; }
.msg-assistant { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 16px; font-size: 14px; line-height: 1.6; }
.msg-loading { color: #dc2626; font-size: 13px; }
.msg-chart-hint { font-size: 11px; color: #6b7280; margin-bottom: 4px; }
.msg-text { white-space: pre-wrap; }
.msg-err { color: #dc2626; }
.chat-input-bar { flex: none; display: flex; align-items: flex-end; gap: 8px; padding: 8px 12px calc(8px + env(safe-area-inset-bottom)); background: #fff; border-top: 1px solid #e5e7eb; }
.chat-input-bar textarea { flex: 1; border: 1px solid #e5e7eb; border-radius: 20px; outline: none; padding: 10px 14px; font-size: 14px; resize: none; max-height: 100px; background: #f9fafb; }
.chat-input-bar textarea:focus { border-color: #dc2626; }
.chat-send { flex: none; padding: 10px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; background: #dc2626; color: #fff; }
.chat-send:disabled { background: #e5e7eb; color: #9ca3af; }
.chat-stop { background: #6b7280; }

/* === 知识库 === */
.muted { color: #9ca3af; }
.kb-upload-btn { display: block; margin: 0 auto; padding: 14px 32px; background: #dc2626; color: #fff; font-size: 15px; font-weight: 700; border-radius: 12px; }
</style>
