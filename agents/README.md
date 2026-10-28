# agents/

## Estado atual

Não há agentes treinados nem código de treinamento nesta pasta. Os únicos agentes
executáveis são os baselines `RandomPolicy` e `GreedyPolicy` em `app/envs/baselines.py`.
O ambiente de jogo completo v2 já existe em `app/envs`, mas ainda não há algoritmo de
treinamento, checkpoint/modelo treinado, experimento de self-play nem benchmark de força.

## Contrato disponível

O ambiente não impõe framework nem algoritmo. O contrato com o motor está em
[`docs/ambiente.md`](../docs/ambiente.md), e o escopo da versão está em
[`docs/release-v1.md`](../docs/release-v1.md). Em resumo:

- v1: ações `Discrete(124)`, observação `Box(427,)` e regras de troca/insolvência avançada desabilitadas.
- v2: ações e observações estruturadas, máscaras condicionais, trocas e insolvência completas.
- `reward_fn` substituível; o padrão é +1 vitória / -1 falência.
- `loop.clone()` para busca/MCTS; `create_game(..., seed=...)` para reprodutibilidade.
- Para avaliar, compare contra `app.envs.GreedyPolicy` e `RandomPolicy` (`python -m app.envs.simulate`).

O motor e os adaptadores v2 aceitam de 2 a 12 jogadores; por padrão, os ambientes v1 usam
até quatro. Para integrar um modelo como oponente no Gymnasium, embrulhe-o como
`policy(loop) -> GameAction` (exemplo no docstring de `app/envs/baselines.py`).

O próximo trabalho aqui é implementar agentes treináveis e experimentos de avaliação.
