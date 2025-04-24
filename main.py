import os, sys, importlib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

# ensure engine & players are importable
sys.path.append(os.path.abspath('.'))

from engine.KuhnPokerEngine import KuhnPokerEngine

app = Flask(
    __name__,
    template_folder='frontend/templates',
    static_folder='frontend/static'
)
app.secret_key = 'replace-with-secure-random-key'

active_game = None
win_counts = [0, 0, 0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/select_mode', methods=['POST'])
def select_mode():
    mode = int(request.form['mode'])
    if mode not in (2, 3):
        return "Invalid mode", 400
    session['mode'] = mode
    return redirect(url_for('select_models'))


@app.route('/select_models')
def select_models():
    mode = session.get('mode')
    if not mode:
        return redirect(url_for('index'))
    files = [
        f[:-3] for f in os.listdir('players')
        if f.endswith('.py') and f != '__init__.py'
    ]
    return render_template('select_models.html',
                           mode=mode,
                           player_modules=files)


@app.route('/start_game', methods=['POST'])
def start_game():
    global active_game, win_counts
    mode = session['mode']
    agents = []
    for i in range(mode):
        mod_name = request.form[f'player{i}']
        mod = importlib.import_module(f'players.{mod_name}')
        cls = next(
            getattr(mod, nm) for nm in dir(mod)
            if isinstance(getattr(mod, nm), type)
               and hasattr(getattr(mod, nm), 'get_action')
        )
        agents.append(cls())

    # reset counts & build engine
    win_counts = [0, 0, 0]
    if mode == 2:
        active_game = KuhnPokerEngine(
            agents[0], agents[1],
            delay=0.0, num_players=2, auto_rounds=None
        )
    else:
        active_game = KuhnPokerEngine(
            agents[0], agents[1], agents[2],
            delay=0.0, num_players=3, auto_rounds=None
        )

    # intercept all engine.log calls into a buffer
    active_game.log_messages = []
    def _log(msg):
        active_game.log_messages.append(msg)
    active_game.log = _log

    return redirect(url_for('game'))


@app.route('/game')
def game():
    if not active_game:
        return redirect(url_for('index'))
    return render_template('game.html')


@app.route('/game_state')
def game_state():
    if not active_game:
        return jsonify(error="No active game"), 400
    return jsonify({
        'pot': active_game.pot,
        'chips': active_game.chips,
        'bets': active_game.current_bets,
        'folded': active_game.folded,
        'cards': active_game.cards,
        'win_counts': win_counts,
        'logs': active_game.log_messages
    })


@app.route('/next_move', methods=['POST'])
def next_move():
    global active_game, win_counts
    if not active_game:
        return jsonify(error="No active game"), 400

    before = list(active_game.chips)
    active_game.run_round()    # run one hand
    after = active_game.chips

    for i, (b, a) in enumerate(zip(before, after)):
        if a > b:
            win_counts[i] += 1

    return jsonify(success=True)


if __name__ == '__main__':
    app.run(debug=True)
