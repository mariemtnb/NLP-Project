'use strict';

const API = '';  // same origin

// Navigation
const navItems = document.querySelectorAll('.nav-item');
const reviewInput = document.getElementById('review-input');

// Health check
async function checkHealth() {
  const dot = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      const data = await res.json();
      dot.className = 'status-dot online';
      label.textContent = `API online · ${data.models_loaded.length} models loaded`;
    } else {
      throw new Error();
    }
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = 'API offline';
  }
}
checkHealth();

// Predict
document.getElementById('predict-btn').addEventListener('click', () => predict());

async function predict() {
  const text = (reviewInput.value || '').trim();
  if (!text) { reviewInput.focus(); return; }

  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analysing...';

  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model: 'svm' }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'API error');
    showResult(data);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">▶</span> Analyse Sentiment';
  }
}

function showResult(data) {
  const { sentiment, confidence, scores } = data;
  const placeholder = document.getElementById('result-placeholder');
  const content = document.getElementById('result-content');
  placeholder.style.display = 'none';
  console.log('Sentiment:', sentiment, 'Confidence:', confidence);
}
