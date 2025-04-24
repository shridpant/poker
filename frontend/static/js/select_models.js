// theme on load
window.onload = () => {
  const t = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', t);
};

document.getElementById('toggleTheme').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const nxt = cur==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme', nxt);
  localStorage.setItem('theme', nxt);
};

// submit
document.getElementById('modelsForm').addEventListener('submit', e => {
  e.preventDefault();
  const data = new FormData(e.target);
  fetch('/start_game',{method:'POST', body:data})
    .then(r => r.redirected && (window.location=r.url))
    .catch(err=>alert(err));
});
