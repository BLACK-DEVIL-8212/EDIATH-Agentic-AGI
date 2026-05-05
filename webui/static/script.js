async function fetchJSON(path, opts) {
  try {
    const res = await fetch(path, opts);
    return await res.json();
  } catch (e) {
    return { error: String(e) };
  }
}

async function refreshStatus() {
  const s = await fetchJSON('/status');
  const el = document.getElementById('status');
  if (s && s.state) {
    el.textContent = `State: ${s.state} | Uptime: ${Math.round(s.uptime_seconds)}s`;
  } else if (s.error) {
    el.textContent = `Error: ${s.error}`;
  } else {
    el.textContent = JSON.stringify(s, null, 2);
  }
}

async function loadAgents() {
  const r = await fetchJSON('/agents');
  const ul = document.getElementById('agents-list');
  ul.innerHTML = '';
  if (r.agents) {
    for (const a of r.agents) {
      const li = document.createElement('li');
      li.textContent = a;
      ul.appendChild(li);
    }
  }
}

document.getElementById('btn-initialize').addEventListener('click', async () => {
  const r = await fetchJSON('/initialize', { method: 'POST' });
  alert(JSON.stringify(r));
  await refreshStatus();
  await loadAgents();
});

document.getElementById('btn-run').addEventListener('click', async () => {
  const r = await fetchJSON('/run', { method: 'POST' });
  alert(JSON.stringify(r));
  await refreshStatus();
});

document.getElementById('btn-shutdown').addEventListener('click', async () => {
  const r = await fetchJSON('/shutdown', { method: 'POST' });
  alert(JSON.stringify(r));
  await refreshStatus();
  await loadAgents();
});

document.getElementById('btn-exec').addEventListener('click', async () => {
  const text = document.getElementById('cmd-text').value;
  const r = await fetchJSON('/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  document.getElementById('exec-result').textContent = JSON.stringify(r, null, 2);
});

// Periodic refresh
setInterval(refreshStatus, 3000);
setInterval(loadAgents, 5000);
refreshStatus();
loadAgents();
