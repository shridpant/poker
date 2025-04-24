// theme on load
window.onload = () => {
  const t = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', t);
};

// toggle
document.getElementById('toggleTheme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const nxt = cur==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme', nxt);
  localStorage.setItem('theme', nxt);
};

// form
document.getElementById('modeForm').addEventListener('submit', e => {
  e.preventDefault();
  const mode = document.querySelector('input[name=mode]:checked').value;
  fetch('/select_mode',{
    method:'POST',
    headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:`mode=${mode}`
  }).then(r=>r.redirected && (window.location=r.url));
});
