# agents/

## Estado atual

Não há checkpoints treinados versionados nesta pasta. Além das baselines
`RandomPolicy` e `GreedyPolicy` em `app/envs/baselines.py`, existe um treinador
experimental PPO mascarado em `app/training/ppo.py`. Ele gera uma política
categórica sobre ações candidatas legais e usa uma rede neural ator-crítico.
Ainda não há self-play nem benchmark concluído de uma política treinada.

## Contrato disponível

O contrato operacional para políticas e treinamento está versionado em
[`docs/contrato-agente-rl-v1.md`](../docs/contrato-agente-rl-v1.md). A especificação
de campos e semântica do estado continua em
[`docs/contrato-observacao-rl-v2.md`](../docs/contrato-observacao-rl-v2.md).
O mapa de camadas e o fluxo entre motor, analytics e adaptadores estão em
[`docs/arquitetura.md`](../docs/arquitetura.md).

Políticas que atuam diretamente no motor seguem o protocolo estrutural
`Agent`: `agent(loop) -> GameAction`. Políticas estocásticas podem implementar
`reset(seed)` (`SeedableAgent`). Para seleção de ações analisadas,
`FeaturePolicy` valida candidatas no motor e `MaxScoreSelector` escolhe a maior
pontuação produzida por um scorer fornecido pelo agente. O projeto não impõe
uma função de utilidade ou pesos estratégicos.

O provedor padrão acrescenta até 32 candidatas de troca legais ao conjunto de
ações enumeradas pelo motor. Ele gera ofertas de dinheiro, ativos individuais,
combinações amostradas de até três ativos por lado e trocas de ativos; não
enumera todas as combinações possíveis. A seed reiniciada em cada episódio
reproduz a amostragem.

Agentes de aprendizado que atuam no Gymnasium implementam
`RLAgent.act(observation, action_mask, deterministic=False)` e devolvem uma
ação válida no espaço do ambiente. `run_gymnasium_episode` executa um episódio, passa a observação e a
máscara ao ator, semeia o ambiente e o agente e retorna retorno acumulado,
número de decisões, término/truncagem e eventual ação ilegal. O ambiente recebe
oponentes como `RandomPolicy` ou `GreedyPolicy`:

```python
from app.envs import MonopolyGymEnvV2, GreedyPolicy, run_gymnasium_episode

env = MonopolyGymEnvV2(opponents=[GreedyPolicy(reserve=150)], learner_seat=0)
result = run_gymnasium_episode(env, agent, seed=2026, deterministic=True)
```

`agent` é o ator implementado pelo consumidor do projeto; esta integração não
escolhe framework ou algoritmo de treinamento.

Para uma política categórica sobre ações legais, `CandidateSelectionEnv` envolve
`MonopolyGymEnvV2` e expõe `Discrete(max_candidates + 1)`. A observação traz
`candidate_actions` (características fixas por linha) e `candidate_mask`; o ator
retorna o índice de uma linha ativa. `candidate_action_mask` inclui também uma
ação extra para solicitar análise de troca.

No modo padrão, `trade_analysis_mode="on_demand"`, esse índice extra calcula
`candidate_decision_features` para propostas e respostas de troca sem avançar a
partida, e o agente decide na observação enriquecida. As features são marcadas
por `candidate_analysis_mask`. O wrapper usa até quatro candidatas e uma
simulação por análise por padrão; os limites podem ser configurados.
A consulta retorna recompensa zero e não avança o estado da partida. A análise
por candidata ainda não cobre trocas durante a insolvência.
`run_gymnasium_episode(..., action_mask_key="candidate_action_mask")` executa
esse formato. Isso evita que o modelo
tenha de gerar diretamente os campos e combinações da ação estruturada.

Para self-play ou uma política por assento, `run_pettingzoo_episode` recebe um
mapa `{agent_name: actor}` e retorna recompensas, decisões e flags de término
por jogador. A observação e sua máscara são entregues apenas ao ator cuja vez é
atual. Agentes com memória recorrente devem usar uma instância por assento;
atores sem estado podem ser compartilhados.

O contrato com o motor está em
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

O treinador PPO em liga, as três heurísticas e o banco de políticas por geração
estão em [`docs/treinamento-ppo.md`](../docs/treinamento-ppo.md). As heurísticas
respondem a ofertas de forma simples, sem avaliar valor nem fazer
contrapropostas; aprender negociação exige oponentes mais completos. Self-play
simultâneo e teoria dos jogos ficam para uma etapa posterior.
