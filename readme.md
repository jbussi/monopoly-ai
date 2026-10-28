# monopoly-ai

Engine de Banco Imobiliário (Monopoly) configurável por JSON, análises estatísticas do
tabuleiro e ambientes para pesquisa em IA. Os ambientes não dependem de algoritmo:
servem para RL (PPO, DQN, AlphaZero/MuZero), busca (MCTS), heurísticas e agentes LLM,
pelas interfaces padrão PettingZoo e Gymnasium.

## Estrutura

```
configs/games/classic_us/   tabuleiro, cartas e regras da variante clássica
app/infraestructure/        leitura e validação das configurações
app/application/            montagem da partida (create_game)
app/domain/game/            engine de regras
app/domain/analytics/       Markov, estatística, valuation
app/envs/                   adaptadores de RL, baselines e simulador
agents/                     espaço reservado a agentes treináveis da v2
data/analytics/             notebooks
docs/ambiente.md            contrato do ambiente: ações, observação, regras, versionamento
docs/release-v1.md          escopo, limitações e checklist da release v1
tests/                      testes de regras, invariantes e conformidade com as APIs
```

O motor aceita partidas de 2 a 12 jogadores. Os adaptadores RL da v1 aceitam 2 a 4,
devido à observação fixa de quatro assentos. A API HTTP e os agentes treinados ainda não
fazem parte desta versão; consulte [escopo da release v1](docs/release-v1.md).

## Instalação

```bash
pip install -e ".[dev]"   # engine + ambientes RL + pytest (Python 3.11+)
pytest -q
```

## Uso rápido

```python
from app.application.game_setup import create_game
loop = create_game(["Ana", "Bia"], seed=0)
while not loop.game.game_over:
    loop.process_action(my_policy(loop))      # loop.legal_actions(), loop.current_decision_player_id

from app.envs import pettingzoo_env           # multiagente
env = pettingzoo_env.env(num_players=4)

from app.envs import MonopolyGymEnv, GreedyPolicy   # agente único
env = MonopolyGymEnv(num_players=2, opponents=[GreedyPolicy()])
```

```bash
python -m app.envs.simulate --games 200 --policies greedy random random random
```
