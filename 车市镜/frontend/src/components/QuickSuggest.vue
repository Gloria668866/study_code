<script setup>
// 追问建议条：根据最近一条助手消息的意图，给出贴合的后续问题（pill）。
import { computed } from 'vue'
const props = defineProps({ messages: { type: Array, default: () => [] } })
const emit = defineEmits(['ask'])

const lastIntent = computed(() => {
  for (let i = props.messages.length - 1; i >= 0; i--) {
    const m = props.messages[i]
    if (m.role === 'assistant' && m.intent) return m.intent
  }
  return 'sql'
})

const suggestions = computed(() => {
  if (lastIntent.value === 'rag') return ['这个结论的依据是什么', '换个角度再解读一下', '有没有相反的观点']
  return ['按月拆开看趋势', '对比去年同期', '还有什么值得注意的']
})
</script>

<template>
  <div class="suggest" v-if="messages.length">
    <span class="lead mono">追问</span>
    <button v-for="s in suggestions" :key="s" class="chip" @click="emit('ask', s)">{{ s }}</button>
  </div>
</template>

<style scoped>
.suggest { max-width: var(--maxw-thread); margin: 0 auto; width: 100%; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 0 24px 2px; }
.lead { font-size: 10px; font-weight: 600; letter-spacing: .1em; color: var(--ink-3); text-transform: uppercase; margin-right: 2px; }
.chip { font-size: 12.5px; color: var(--ink-2); padding: 6px 13px; border: 1px solid var(--line); background: var(--bg); border-radius: 999px; cursor: pointer; transition: border-color .12s, color .12s, background .12s; }
.chip:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-wash); }
</style>
