<script setup>
import { ref } from 'vue'
import { useAuth } from '@/composables/useAuth.js'
import LogoMark from './LogoMark.vue'

const { login, register, state } = useAuth()
const mode = ref('login')
const form = ref({ username: '', password: '', nickname: '' })
const busy = ref(false)
async function submit() { busy.value = true; const fn = mode.value === 'login' ? login : register; await fn({ ...form.value }); busy.value = false }
</script>

<template>
  <div class="login">
    <div class="bg">
      <div class="grid"></div>
      <div class="glow"></div>
    </div>

    <div class="panel">
      <div class="brand">
        <div class="mark"><LogoMark :size="40" tone="brand" /></div>
        <h1>车<span>市</span>镜</h1>
        <p class="tag mono">EV-MARKETLENS · 新能源车市情报终端</p>
      </div>

      <div class="card">
        <div class="tabs">
          <button type="button" :class="{ on: mode === 'login' }" @click="mode = 'login'">登录</button>
          <button type="button" :class="{ on: mode === 'register' }" @click="mode = 'register'">注册</button>
        </div>
        <form @submit.prevent="submit">
          <div v-if="state.error" class="err">{{ state.error }}</div>
          <label class="field"><span>用户名</span><input v-model="form.username" placeholder="用户名" required autocomplete="username" /></label>
          <label class="field"><span>密码</span><input v-model="form.password" type="password" placeholder="密码" required :autocomplete="mode === 'login' ? 'current-password' : 'new-password'" /></label>
          <label v-if="mode === 'register'" class="field"><span>昵称</span><input v-model="form.nickname" placeholder="昵称（选填）" /></label>
          <button class="submit" :disabled="busy">{{ busy ? '处理中…' : mode === 'login' ? '进入工作台' : '创建账号' }}</button>
        </form>
        <p class="hint">双脑 Agent · Text2SQL 查库出图 · RAG 读研报带引用</p>
      </div>

      <p class="foot">覆盖 101 品牌 · 409 车系 · 8,072 条市场数据</p>
    </div>
  </div>
</template>

<style scoped>
.login { height: 100%; display: flex; align-items: center; justify-content: center; background: var(--bg-subtle); position: relative; overflow: hidden; padding: 20px; }
.bg { position: absolute; inset: 0; pointer-events: none; }
.grid { position: absolute; inset: 0; opacity: .5; background-image: linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px); background-size: 44px 44px; -webkit-mask-image: radial-gradient(circle at 50% 42%, #000, transparent 70%); mask-image: radial-gradient(circle at 50% 42%, #000, transparent 70%); }
.glow { position: absolute; top: -10%; left: 50%; transform: translateX(-50%); width: 620px; height: 420px; background: radial-gradient(circle, rgba(220,38,38,.10), transparent 62%); }

.panel { position: relative; z-index: 1; width: 400px; max-width: 94vw; animation: fadeUp .5s ease both; }
.brand { text-align: center; margin-bottom: 22px; }
.mark { display: inline-flex; padding: 14px; background: var(--bg); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--sh-md); }
h1 { font-family: var(--font-display); font-size: 30px; font-weight: 800; margin: 16px 0 0; letter-spacing: .04em; color: var(--ink); }
h1 span { color: var(--accent); }
.tag { font-size: 10.5px; color: var(--ink-3); letter-spacing: .14em; margin: 8px 0 0; }

.card { background: var(--bg); border: 1px solid var(--line); border-radius: var(--r-lg); box-shadow: var(--sh-md); padding: 24px; }
.tabs { display: flex; gap: 3px; background: var(--bg-sunken); padding: 4px; border-radius: var(--r-md); margin-bottom: 18px; }
.tabs button { flex: 1; padding: 9px; font-size: 13.5px; font-weight: 700; color: var(--ink-3); border-radius: 9px; transition: .12s; }
.tabs button.on { background: var(--bg); color: var(--ink); box-shadow: var(--sh-sm); }
.err { font-size: 12.5px; color: var(--accent); background: var(--accent-wash); border: 1px solid var(--accent-border); padding: 9px 12px; border-radius: var(--r-sm); margin-bottom: 14px; }
.field { display: block; margin-bottom: 12px; }
.field span { display: block; font-family: var(--font-mono); font-size: 10.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 5px; }
.field input { width: 100%; padding: 11px 13px; font-size: 14px; background: var(--bg-subtle); border: 1px solid var(--line); border-radius: var(--r-sm); outline: none; transition: .15s; }
.field input:focus { border-color: var(--accent); background: var(--bg); box-shadow: 0 0 0 3px var(--accent-wash); }
.field input::placeholder { color: var(--ink-4); }
.submit { width: 100%; margin-top: 6px; padding: 12px; font-size: 14px; font-weight: 700; color: #fff; background: var(--accent); border-radius: var(--r-md); transition: background .12s; }
.submit:hover:not(:disabled) { background: var(--accent-press); }
.submit:disabled { opacity: .55; cursor: not-allowed; }
.hint { text-align: center; margin: 16px 0 0; font-size: 12px; color: var(--ink-3); }
.foot { text-align: center; margin: 20px 0 0; font-family: var(--font-mono); font-size: 11px; color: var(--ink-3); letter-spacing: .03em; }
</style>
