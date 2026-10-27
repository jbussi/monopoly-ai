# agents/

Área livre para treino, modelos e avaliação. O ambiente não impõe framework nem algoritmo.

O contrato com o ambiente está em [`docs/ambiente.md`](../docs/ambiente.md). Resumo:

- Ações `Discrete(124)`, sempre com máscara (`obs["action_mask"]` no PettingZoo, `env.action_masks()` no Gymnasium).
- Observação `Box(427,)` float32, com o slot 0 sendo o próprio agente.
- `reward_fn` substituível; o padrão é +1 vitória / -1 falência.
- `loop.clone()` para busca/MCTS; `create_game(..., seed=...)` para reprodutibilidade.
- Para avaliar, compare contra `app.envs.GreedyPolicy` e `RandomPolicy` (`python -m app.envs.simulate`).

Para usar um modelo como oponente, embrulhe-o como `policy(loop) -> GameAction`
(exemplo no docstring de `app/envs/baselines.py`).
