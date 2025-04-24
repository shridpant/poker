let timer = null, positions=['seat-top','seat-left','seat-right'];

function fetchState(){
  fetch('/game_state')
    .then(r=>r.json())
    .then(d=>{
      // pot & chips
      document.getElementById('pot').textContent = d.pot;
      d.chips.forEach((c,i)=>{
        document.querySelector(`.${positions[i]} .name`)
                .textContent = `P${i+1}: ${c}`;
      });
      // win counts
      document.getElementById('w1').textContent = d.win_counts[0];
      document.getElementById('w2').textContent = d.win_counts[1];
      document.getElementById('w3').textContent = d.win_counts[2];
      // cards
      d.cards.forEach((c,i)=>{
        const div = document.querySelector(`.${positions[i]} .cards`);
        div.innerHTML = `<div class="card">${c}</div>`;
      });
      // logs
      const logDiv = document.getElementById('gameLog');
      logDiv.innerHTML = '';
      d.logs.forEach(line=>{
        const e = document.createElement('div');
        e.textContent = line;
        logDiv.appendChild(e);
      });
    });
}

function step(){
  fetch('/next_move',{method:'POST'})
    .then(()=> fetchState());
}

// theme persist
window.onload = ()=>{  
  const t = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme',t);
  fetchState();
};
document.getElementById('toggleTheme').onclick = ()=>{
  const cur = document.documentElement.getAttribute('data-theme');
  const nxt = cur==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme', nxt);
  localStorage.setItem('theme', nxt);
};

// controls
document.getElementById('startSim').onclick = ()=>{
  const interval = +document.getElementById('speedControl').value;
  timer = setInterval(step, interval);
  document.getElementById('startSim').style.display = 'none';
  document.getElementById('stopSim').style.display  = 'inline-block';
};
document.getElementById('stopSim').onclick = ()=>{
  clearInterval(timer);
  timer = null;
  document.getElementById('stopSim').style.display  = 'none';
  document.getElementById('startSim').style.display = 'inline-block';
};
