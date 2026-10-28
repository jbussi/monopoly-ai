# agents/

## Estado na v1

Não há agentes treinados nem código de treinamento nesta pasta. Os únicos agentes
executáveis são os baselines `RandomPolicy` e `GreedyPolicy` em `app/envs/baselines.py`.
Esta pasta fica reservada para os agentes e experimentos planejados para a v2.

## Contrato disponível

O ambiente não impõe framework nem algoritmo. O contrato com o motor está em
[`docs/ambiente.md`](../docs/ambiente.md), e o escopo da versão está em
[`docs/release-v1.md`](../docs/release-v1.md). Em resumo:

- Ações `Discrete(124)`, sempre com máscara (`obs["action_mask"]` no PettingZoo, `env.action_masks()` no Gymnasium).
- Observação `Box(427,)` float32, com o slot 0 sendo o próprio agente.
- `reward_fn` substituível; o padrão é +1 vitória / -1 falência.
- `loop.clone()` para busca/MCTS; `create_game(..., seed=...)` para reprodutibilidade.
- Para avaliar, compare contra `app.envs.GreedyPolicy` e `RandomPolicy` (`python -m app.envs.simulate`).

O motor aceita 2 a 12 jogadores; os adaptadores RL atuais usam quatro assentos e aceitam
2 a 4. Para integrar um modelo como oponente, embrulhe-o como
`policy(loop) -> GameAction` (exemplo no docstring de `app/envs/baselines.py`).

A v2 está planejada para incluir o código dos agentes treináveis e os experimentos de
avaliação, junto com as expansões de regras descritas no documento de release.
