/* ── CONFIG ─────────────────────────────────── */
const WS_URL = (() => {
  const p = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return p + '//' + location.host + '/ws';
})();

/* ── STATE ─────────────────────────────────── */
let ws;
let currentScreen = 'dashboard';
let chatHistory = [];

/* ── HELPERS ────────────────────────────────── */
function esc(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function ts() { return Date.now() / 1000; }

/* ── NAVIGATION ──────────────────────────────── */
function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const s = document.getElementById('screen-' + name);
  if (s) s.classList.add('active');
  const btn = document.querySelector('.nav-item[data-screen="' + name + '"]');
  if (btn) btn.classList.add('active');
  currentScreen = name;
  if (name === 'dashboard') refreshDashboard();
  if (name === 'chat') scrollChat();
}

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => showScreen(btn.dataset.screen));
});

/* ── WEBSOCKET ──────────────────────────────── */
function connect() {
  setStatus('Connecting...');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus('Connected', 'online');
  };

  ws.onmessage = (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }
    if (!data || !data.type) return;

    if (data.type === 'history') {
      chatHistory = data.messages || [];
      renderHistory();
      return;
    }

    if (data.type === 'message') {
      const m = data.message;
      if (m) {
        chatHistory.push(m);
        appendMessage(m.role, m.content, m.ts);
        if (currentScreen !== 'chat') {
          // Badge on nav
          const chatBtn = document.querySelector('.nav-item[data-screen="chat"]');
          if (chatBtn) chatBtn.style.setProperty('--new', '•');
        }
      }
      return;
    }

    if (data.type === 'status') {
      updateDashboardStatus(data);
      return;
    }
  };

  ws.onclose = () => {
    setStatus('Disconnected — reconnecting...', 'error');
    setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    setStatus('WebSocket error', 'error');
  };
}

function sendRaw(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function setStatus(text, state) {
  const el = document.getElementById('connectionStatus');
  const dot = document.getElementById('statusDot');
  if (el) el.textContent = text;
  if (dot) {
    dot.className = 'status-dot' + (state ? ' ' + state : '');
  }
}

/* ── CHAT ───────────────────────────────────── */
const messagesEl = document.getElementById('messages');
const composer = document.getElementById('composer');
const inputEl = document.getElementById('input');

function addMsg(role, content, ts, silent) {
  if (!messagesEl) return;
  const row = document.createElement('div');
  row.className = 'row ' + (role === 'YOU' ? 'you' : 'ai');

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const who = role === 'YOU' ? 'You' : role === 'SYSTEM' ? 'System' : 'AI';
  const time = ts ? new Date(ts * 1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) : '';

  bubble.innerHTML = `
    <div class="who">${esc(who)} ${time ? '<span class="time">'+esc(time)+'</span>' : ''}</div>
    <div class="text">${esc(content).replaceAll('\n', '<br/>')}</div>
  `;

  row.appendChild(bubble);
  messagesEl.appendChild(row);
  if (!silent) scrollChat();
}

function renderHistory() {
  if (messagesEl) messagesEl.innerHTML = '';
  (chatHistory || []).forEach(m => addMsg(m.role, m.content, m.ts, true));
  scrollChat();
}

function scrollChat() {
  if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function clearChat() {
  chatHistory = [];
  if (messagesEl) messagesEl.innerHTML = '';
  sendRaw({type: 'clear_history'});
}

/* ── CHAT FORM ──────────────────────────────── */
if (composer) {
  composer.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;

    addMsg('YOU', text, ts());
    inputEl.value = '';
    showScreen('chat');

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({text})
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        addMsg('SYSTEM', body.error || 'HTTP ' + res.status, ts());
      }
    } catch(err) {
      addMsg('SYSTEM', 'Request failed: ' + err.message, ts());
    }
  });
}

const btnClearChat = document.getElementById('btnClearChat');
if (btnClearChat) btnClearChat.addEventListener('click', clearChat);

/* ── DASHBOARD ──────────────────────────────── */
function updateDashboardStatus(data) {
  const statusEl = document.getElementById('sysStatus');
  const llmEl = document.getElementById('llmStatus');
  if (statusEl && data.status) statusEl.textContent = data.status;
  if (llmEl && data.llm_ready) {
    llmEl.textContent = data.llm_ready ? 'LLM Ready' : 'LLM Loading...';
    llmEl.className = 'badge ' + (data.llm_ready ? 'ok' : '');
  }
}

function refreshDashboard() {
  // Simulated stats — in a real app these come from the backend via /api/stats
  const uptimeEl = document.getElementById('uptime');
  if (uptimeEl) {
    const s = Math.floor((Date.now() - window._startTime) / 1000);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    uptimeEl.textContent = h > 0 ? h + 'h ' + m + 'm' : m + 'm';
  }
  sendRaw({type: 'get_stats'});
}

window._startTime = Date.now();
setInterval(() => {
  if (currentScreen === 'dashboard') refreshDashboard();
}, 15000);

/* ── QUICK ACTIONS ──────────────────────────── */
const btnNewChat = document.getElementById('btnNewChat');
if (btnNewChat) btnNewChat.addEventListener('click', () => { clearChat(); showScreen('chat'); });

const btnExplore = document.getElementById('btnExplore');
if (btnExplore) btnExplore.addEventListener('click', async () => {
  try {
    await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text: 'Explore my recent memories and summarize what you found.'})
    });
    showScreen('chat');
  } catch(e) {}
});

const btnSchedule = document.getElementById('btnSchedule');
if (btnSchedule) btnSchedule.addEventListener('click', () => showScreen('tasks'));

/* ── TASKS ─────────────────────────────────── */
const btnRefreshTasks = document.getElementById('btnRefreshTasks');
if (btnRefreshTasks) btnRefreshTasks.addEventListener('click', loadTasks);

async function loadTasks() {
  const el = document.getElementById('taskList');
  if (!el) return;
  try {
    const res = await fetch('/api/tasks');
    if (res.ok) {
      const tasks = await res.json();
      el.innerHTML = tasks.length
        ? tasks.map(t => `<div class="task-item">
            <span class="task-title">${esc(t.title || t.id)}</span>
            <span class="task-status ${esc(t.status)}">${esc(t.status)}</span>
          </div>`).join('')
        : '<div class="card">No active tasks.</div>';
    } else {
      el.innerHTML = '<div class="card">Could not load tasks.</div>';
    }
  } catch {
    el.innerHTML = '<div class="card">Task queue unavailable.</div>';
  }
}

/* ── INIT ──────────────────────────────────── */
connect();

const dashBtn = document.querySelector('.nav-item[data-screen="dashboard"]');
if (dashBtn) dashBtn.addEventListener('click', refreshDashboard);

refreshDashboard();