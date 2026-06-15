import { reactive, computed } from 'vue'
import { API, getToken, setToken, clearToken, getUser, setUser, authHeaders } from '@/api/config.js'

const state = reactive({
  user: getUser(),
  token: getToken(),
  error: '',
  pending: false,
})

export const isAuthed = computed(() => !!state.token && !!state.user)

async function login(username, password) {
  state.pending = true; state.error = ''
  try {
    const r = await fetch(API.login, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    const d = await r.json()
    if (!r.ok) throw new Error(d.detail || '登录失败')
    setToken(d.access_token); setUser(d.user)
    state.token = d.access_token; state.user = d.user
  } catch (e) { state.error = e.message } finally { state.pending = false }
}

async function register(username, password, nickname) {
  state.pending = true; state.error = ''
  try {
    const r = await fetch(API.register, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, nickname }),
    })
    const d = await r.json()
    if (!r.ok) throw new Error(d.detail || '注册失败')
    // 注册后自动登录需要再调一次 login
    await login(username, password)
  } catch (e) { state.error = e.message } finally { state.pending = false }
}

function logout() { clearToken(); state.token = null; state.user = null }

export function useAuth() { return { state, isAuthed, login, register, logout } }
