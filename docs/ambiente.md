# Ambiente de RL `monopoly_v1`

Este documento é o **contrato** entre a engine (`app/domain/game`) e o código de treino.
Dentro da v1, tudo o que está aqui é estável. Mudar um tamanho ou uma ordem invalida
modelos treinados e exige uma v2 (ver [Versionamento](#versionamento)).

## Camadas

```
configs/games/<variante>/      board.json, cards.json, settings.json
app/infraestructure/game_config  lê e valida os JSONs        -> GameConfig
app/application/game_setup       monta a partida              -> create_game(...) -> GameLoop
app/domain/game                  regras (GameLoop, GameState): ações em texto, sem IA
app/envs                         adaptadores: espaço discreto, observação, PettingZoo, Gymnasium, texto
agents/                          treino e modelos (área livre)
```

A engine não depende de numpy, gymnasium nem pettingzoo. A API dela é
`GameLoop.process_action(GameAction)`, `legal_actions()` e `validate(action)`, as mesmas
que uma API web ou um humano usariam. As duas primeiras usam a mesma validação, então a
máscara de ações nunca diverge do que a engine aceita (há teste para isso).

## Regras

Regras clássicas americanas, lidas do `settings.json`:

- compra de imóvel e **leilão** do imóvel recusado (`auctionEnabled`);
- aluguel com valor de monopólio do `board.json`, casas e hotel;
- ferrovias (25/50/100/200) e companhias (4x/10x os dados);
- Income Tax: o menor valor entre $200 e 10% do patrimônio (`incomeTaxChoice`);
- as 32 cartas do `cards.json`, executadas pelo `CardActionRegistry`;
- prisão: 3 duplos, a casa "Go to Jail" ou carta; para sair, duplo, multa, carta ou multa obrigatória na 3ª tentativa;
- rolagem extra com duplo;
- construção uniforme e estoque de 32 casas e 12 hotéis;
- hipoteca e custo para levantá-la (`unmortgageCost`);
- resolução de dívida (vender construções ou hipotecar) e falência.

Simplificações da v1:

| Regra | v1 |
|---|---|
| Negociação entre jogadores (`tradeEnabled`) | **Não implementada** |
| Gerenciar imóveis fora do próprio turno | Só como devedor (`RESOLVE_DEBT`) |
| Levantar dinheiro durante um leilão | Não; o lance é limitado ao dinheiro em caixa |
| Falência com o banco como credor | Imóveis voltam ao banco, sem leilão |
| Falência com vários credores (ex.: "pague $50 a cada jogador") | Bens vão para o banco |
| Juros de 10% ao herdar imóvel hipotecado | Não cobrados |
| Desfazer hotel sem 4 casas no banco | Proibido |
| `freeParkingJackpot`, `bankruptcyToBank`, `auctionTimeoutSeconds` | Ignorados pela engine |

**Truncamento:** com `max_turns` (padrão 1000 nos ambientes), a partida termina sem
vencedor. Sem negociação, monopólios são raros: entre 4 jogadores `GreedyPolicy`, cerca
de 77% das partidas chegam ao limite; com 2 jogadores, cerca de 12%.

## Fases e quem decide

`loop.current_decision_player_id` é sempre quem decide. Pode não ser o dono do turno:
num leilão é quem tem a vez de dar lance; numa dívida, é o devedor (ex.: carta de aniversário).

| Fase | Ações |
|---|---|
| `BEFORE_ROLL` | `ROLL_DICE`; na prisão também `PAY_JAIL_FINE` e `USE_JAIL_CARD`; gerenciamento |
| `AWAITING_PROPERTY_DECISION` | `BUY_PROPERTY` (se tiver dinheiro), `DECLINE_PROPERTY`; gerenciamento |
| `AUCTION` | `AUCTION_BID {amount}`, `AUCTION_WITHDRAW` |
| `AFTER_ROLL` | `END_TURN`; gerenciamento |
| `RESOLVE_DEBT` | `SELL_HOUSE`, `MORTGAGE`, `DECLARE_BANKRUPTCY` |
| `GAME_OVER` | nenhuma |

"Gerenciamento" = `BUILD_HOUSE`, `SELL_HOUSE`, `MORTGAGE` e `UNMORTGAGE`, todas com `{tile_id}`.
Um turno tem várias decisões do mesmo agente. **Sempre há pelo menos uma ação legal**
enquanto a partida não acaba. Uma dívida é paga automaticamente assim que o devedor tem
o dinheiro; se nem vendendo tudo ele consegue pagar, a falência é imediata.

## Espaço de ações: `Discrete(124)`

P = 28 imóveis, na ordem do tabuleiro (índice 0 = Mediterranean, posição 1; índice 27 = Boardwalk, posição 39).

| Índices | Ação |
|---|---|
| 0 | `ROLL_DICE` |
| 1 | `BUY_PROPERTY` |
| 2 | `DECLINE_PROPERTY` |
| 3 | `END_TURN` |
| 4 | `PAY_JAIL_FINE` |
| 5 | `USE_JAIL_CARD` |
| 6 | `DECLARE_BANKRUPTCY` |
| 7 | `AUCTION_WITHDRAW` |
| 8 – 11 | `AUCTION_BID` de +1, +10, +50 ou +100 sobre o maior lance (mínimo: `auctionMinBid`) |
| 12 – 39 | `BUILD_HOUSE[p]` |
| 40 – 67 | `SELL_HOUSE[p]` |
| 68 – 95 | `MORTGAGE[p]` |
| 96 – 123 | `UNMORTGAGE[p]` |

Use `DiscreteActionSpace.to_game_action(i, loop)`, `.to_index(action, loop)` e `.mask(loop)`,
sem fixar índices no código. A engine aceita qualquer lance inteiro válido; os 4 incrementos
são só a discretização oferecida aos agentes. Para mudá-la, altere `AUCTION_BID_INCREMENTS`
(isso muda o tamanho do espaço, ou seja, é uma nova versão).

## Observação: `Box(427,)` float32

Sempre do ponto de vista do observador, com os jogadores rotacionados: o **slot 0 é o
observador**, o slot 1 é o próximo na ordem da mesa, e assim por diante. Há sempre 4 slots;
os vazios ou falidos ficam zerados. Assim o mesmo modelo serve para 2, 3 ou 4 jogadores.

| Faixa | Conteúdo |
|---|---|
| 0 – 183 | 4 slots × 46: `presente`, `dinheiro/1500`, `posição` one-hot (40), `preso`, `tentativas_prisão/3`, `cartas_prisão/2`, `é_dono_do_turno` |
| 184 – 379 | 28 imóveis × 7: dono one-hot `[banco, slot0..slot3]`, `casas/5` (hotel = 1), `hipotecado` |
| 380 – 385 | fase one-hot (`BEFORE_ROLL`, `AWAITING_PROPERTY_DECISION`, `AUCTION`, `AFTER_ROLL`, `RESOLVE_DEBT`, `GAME_OVER`) |
| 386 – 413 | imóvel em decisão de compra ou em leilão (one-hot) |
| 414 – 419 | leilão: `maior_lance/1500`, maior licitante one-hot `[nenhum, slot0..slot3]` |
| 420 – 426 | `duplos/3`, `dívida_do_observador/1500`, `casas_no_banco/32`, `hotéis_no_banco/12`, `turno/max_turns`, `dado1/6`, `dado2/6` |

Os offsets exatos estão em `ObservationEncoder.layout`. Dados estáticos (preços, aluguéis)
**não** entram na observação, porque são constantes da variante. Quem quiser pode
concatená-los ou escrever outro encoder: o encoder fica fora da engine justamente para isso.

**Informação oculta:** a ordem dos baralhos não aparece na observação. `loop.clone()`
copia tudo, inclusive RNGs e baralhos. Numa busca, para não "trapacear", embaralhe os
baralhos do clone (*determinization*).

## Recompensa

Padrão: `win_loss_reward`, que dá +1 ao vencedor, -1 a quem vai à falência e 0 no resto
(inclusive na truncagem). Para trocar, passe `reward_fn=lambda loop, player_id: ...`.
A função é chamada após cada ação, para cada jogador ativo. `GameState` expõe `net_worth`,
`liquidation_value` e `rent_for` para montar o shaping. As análises de `app/domain/analytics`
(distribuição estacionária, ROI por cor) também servem de base para shaping ou para features extras.

## Interfaces

| Interface | Para quê | Como |
|---|---|---|
| Engine | busca/MCTS, LLM, API, qualquer coisa custom | `create_game(...)`, `legal_actions()`, `process_action()`, `clone()` |
| PettingZoo AEC | multiagente / self-play (RLlib, Tianshou, TorchRL…) | `app.envs.pettingzoo_env.env(num_players=4)` |
| Gymnasium | agente único contra oponentes fixos (SB3, `MaskablePPO`, CleanRL) | `MonopolyGymEnv(num_players, opponents=[...])` |
| Texto | render e agentes LLM | `render_text(loop)`, `describe_action(loop, action)` |

Uma política é qualquer callable `policy(loop) -> GameAction`. `RandomPolicy` e
`GreedyPolicy` são as referências: um agente treinado deve vencer a `GreedyPolicy` com folga.

## Reprodutibilidade

- `create_game(..., seed=s)` controla os dados, os baralhos e os IDs dos jogadores. A mesma seed com as mesmas ações reproduz a mesma partida.
- `env.reset(seed=s)` também semeia os oponentes que têm `reset(seed)`.
- `record_events=True` grava a telemetria em `game.history_logs`. Os ambientes desligam isso por desempenho.
- Velocidade atual: cerca de 8 a 10 mil ações por segundo em um núcleo. O gargalo é o `legal_actions()`.

## Versionamento

O ambiente se chama `monopoly_v1` (`app.envs.ENV_VERSION`). Expansões como a negociação
mudam o espaço de ações e/ou de observação e entram como **v2**, com estas regras:

1. Blocos de ação novos só são **acrescentados no final**.
2. Campos novos de observação entram **no final** do vetor.
3. Regras novas ficam atrás de flags (`settings.json` / `Rules`), e a v1 continua disponível.
