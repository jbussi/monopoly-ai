# Contratos dos ambientes e regras do motor

Este documento descreve o motor e os contratos dos ambientes `monopoly_v1` e
`monopoly_v2`. Use a [arquitetura do projeto](arquitetura.md) para navegar entre
os pacotes, o [contrato de agente](contrato-agente-rl-v1.md) para integrar uma
política e o [contrato de observação](contrato-observacao-rl-v2.md) para
entender os campos do ambiente v2.

O motor aceita partidas com 2 a 12 jogadores. Os adaptadores legados v1 têm uma
observação fixa para até quatro jogadores; os ambientes v2 aceitam de 2 a 12.

## Camadas

```
configs/games/<variante>/      board.json, cards.json, settings.json
app/infraestructure/game_config  lê e valida os JSONs        -> GameConfig
app/application/game_setup       monta a partida              -> create_game(...) -> GameLoop
app/domain/game                  regras (GameLoop, GameState): ações em texto, sem IA
app/envs                         adaptadores: espaço discreto, observação, PettingZoo, Gymnasium, texto
app/domain/agents                protocolos e adaptadores de políticas
agents/                          guia do estado dos agentes e do treino
```

A engine de domínio não depende de Gymnasium nem PettingZoo. Sua interface é
`GameLoop.process_action(GameAction)`, `legal_actions()` e `validate(action)`, as mesmas
que uma API web ou um humano usariam. As duas primeiras usam a mesma validação, então a
máscara de ações corresponde às ações aceitas (há teste para isso).

## Regras

Regras clássicas americanas. `settings.json` é a fonte oficial para valores e flags de
regras; `board.json` define a estrutura do tabuleiro, os grupos e as tabelas de aluguel.
Quando um valor aparece nos dois arquivos, o motor usa `settings.json`.

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

Simplificações do motor e escopo do contrato RL `monopoly_v1`:

| Regra | v1 |
|---|---|
| Negociação entre jogadores (`tradeEnabled`) | Disponível no motor; o contrato RL `monopoly_v1` mantém negociação desligada |
| Gerenciar imóveis fora do próprio turno | Só como devedor (`RESOLVE_DEBT`) |
| Levantar dinheiro durante um leilão | Não; o lance é limitado ao dinheiro em caixa |
| Destino dos bens na falência (`bankruptcyToBank`) | Quando ativo, o dinheiro restante vai ao banco e os imóveis retornam sem hipoteca e são leiloados; é o comportamento configurado em `classic_us` |
| Falência com vários credores (ex.: "pague $50 a cada jogador") | Bens vão para o banco |
| Imóvel hipotecado transferido | O novo dono escolhe pagar 10% para manter a hipoteca ou resgatá-la imediatamente por `unmortgageCost` |
| Vender hotel sem 4 casas no banco | Permitido; registra escassez e compensa devoluções futuras |
| `auctionTimeoutSeconds` | Ignorado pela engine |

**Truncamento:** com `max_turns` (padrão 1000 nos ambientes), a partida termina sem
vencedor. Sem negociação, monopólios são raros: entre 4 jogadores `GreedyPolicy`, cerca
de 77% das partidas chegam ao limite; com 2 jogadores, cerca de 12%.

## Fases e quem decide

`loop.current_decision_player_id` é sempre quem decide. Pode não ser o dono do turno:
num leilão é quem tem a vez de dar lance; numa dívida, é o devedor (ex.: carta de aniversário).

| Fase | Ações |
|---|---|
| `BEFORE_ROLL` | janela de negociação (se habilitada); `ROLL_DICE`; na prisão também `PAY_JAIL_FINE` e `USE_JAIL_CARD`; gerenciamento |
| `AWAITING_PROPERTY_DECISION` | `BUY_PROPERTY` (se tiver dinheiro), `DECLINE_PROPERTY`; gerenciamento |
| `AUCTION` | `AUCTION_BID {amount}`, `AUCTION_WITHDRAW` |
| `AFTER_ROLL` | `END_TURN`; gerenciamento |
| `RESOLVE_DEBT` | `SELL_HOUSE`, `MORTGAGE`, `PROPOSE_TRADE`/respostas, `END_DEBT_NEGOTIATIONS`, `AUCTION_DEBT_PROPERTY`, `DECLARE_BANKRUPTCY` |
| `RESOLVE_MORTGAGE_TRANSFER` | O recebedor escolhe manter a hipoteca pagando 10% ou resgatá-la imediatamente pelo `unmortgageCost` |
| `GAME_OVER` | nenhuma |

"Gerenciamento" = `BUILD_HOUSE`, `SELL_HOUSE`, `MORTGAGE` e `UNMORTGAGE`, todas com `{tile_id}`.
Um turno pode conter várias decisões do mesmo agente. Quando o caixa cobre a dívida,
ela é paga automaticamente. Caso contrário, o fluxo permite vender construções,
hipotecar imóveis, negociar com credores dentro dos limites configurados e, após
encerrar negociações, leiloar propriedades sem construções. A falência é declarada
quando as opções permitidas de recuperação se esgotam.

### Negociações no motor

Com `tradeEnabled: true`, cada turno abre uma janela antes dos dados. O jogador da vez pode
enviar uma oferta estruturada com dinheiro, propriedades e cartas de saída da prisão; o
destinatário responde aceitando, rejeitando ou contrapondo. A janela pausa até a resposta,
e a rolagem encerra a negociação daquele turno. O limite padrão é de três contrapropostas
após a oferta inicial (`tradeMaxCounteroffers` em `settings.json`). O máximo de uma janela
por turno limita o volume de novas ofertas; só pode haver uma negociação pendente.

Aceitar revalida todos os itens e transfere dinheiro, propriedades e cartas numa única
operação. Se uma propriedade hipotecada for transferida, o novo dono decide imediatamente
entre mantê-la hipotecada pagando 10% do valor hipotecário ao banco, ou resgatá-la pagando o
`unmortgageCost`. A negociação pausa até essa decisão; se os juros não puderem ser pagos,
abre-se o fluxo normal de dívidas. Não se pode negociar uma propriedade
enquanto houver construções em qualquer imóvel do seu grupo de cor; primeiro, todas as
construções do grupo devem ser vendidas ao banco.

As ofertas e contrapropostas usam ações estruturadas `GameAction`, cujos termos não cabem
no espaço discreto fixo `monopoly_v1`. Os adaptadores v1 mantêm as trocas desabilitadas;
os adaptadores v2 descritos abaixo expõem esses comandos em ações estruturadas. No motor,
`available_trade_actions()` informa os comandos
estruturados disponíveis, e `trade_response_actions()` enumera aceitar, rejeitar e
contrapropor; `validate()` e `process_action()` verificam os termos concretos.

Na liquidação, um hotel pode ser vendido mesmo quando o banco não tem quatro casas físicas
disponíveis. O motor registra essa falta em `GameState.house_shortage`; casas devolvidas
posteriormente primeiro compensam a escassez, antes de aumentar o estoque físico do banco.
Assim a venda do hotel não fica bloqueada e o estoque nunca fica negativo.

Quando uma dívida é aberta, o jogador pode vender construções e hipotecar imóveis ao banco,
negociar com até `debtNegotiationsPerPlayer` jogadores distintos (configuração atual: 3),
encerrando explicitamente essa etapa, e então leiloar propriedades sem construções. Cada
leilão exclui o devedor, e o valor vencedor é creditado a ele para quitar as dívidas na ordem
em que surgiram. A falência só fica disponível depois do encerramento das negociações e
quando não restam ações bancárias nem imóveis elegíveis para leilão. Com `bankruptcyToBank`
ativo, na falência o dinheiro restante é entregue ao banco e os imóveis restantes retornam
sem hipoteca e são leiloados individualmente aos jogadores ativos. O dinheiro não é leiloado;
ele é entregue ao banco. A observação v2 identifica os leilões de imóveis em
`auction.is_bankruptcy_auction`. Cada oferta pendente
interrompe a partida até ser aceita, rejeitada ou respondida com contraproposta, limitada por
`tradeMaxCounteroffers`. Os adaptadores v1 mantêm o espaço discreto legado e desabilitam as
funcionalidades avançadas; os adaptadores v2 as expõem.

O vetor `rules` da observação v2 termina com `bankruptcyToBank` e a taxa de juros de
transferência hipotecária (0.10), nessa ordem.

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
- Não há benchmark de desempenho automatizado mantido como parte do contrato v1.

## Versionamento

O ambiente legado se chama `monopoly_v1` (`app.envs.ENV_VERSION`). O contrato estruturado
completo se chama `monopoly_v2` (`app.envs.ENV_VERSION_V2`). Mudanças futuras que afetem
espaço de ações ou observações devem seguir estas regras:

1. No contrato discreto v1, novos blocos de ação só são acrescentados ao final;
   não se reordenam nem se reaproveitam índices publicados.
2. Na observação vetorial v1, novos campos só são acrescentados ao final. Na v2,
   campos são nomeados e devem manter semântica e forma; adições são feitas como
   chaves novas ou mudanças versionadas do espaço.
3. Regras que alteram partidas são configuradas em `settings.json` e modeladas
   em `Rules`; o adaptador v1 permanece disponível para consumidores antigos.

## Ambiente v2: regras completas e ações estruturadas

`MonopolyGymEnvV2` (`app.envs.gymnasium_env_v2`) e `pettingzoo_env_v2.env()` usam a
configuração integral da engine, inclusive trocas e resolução avançada de insolvência.
Aceitam partidas de 2 a 12 jogadores (`max_players` também determina o tamanho fixo dos
vetores de jogadores). A v1 permanece disponível e continua usando `Discrete(124)`.

A observação v2 é um `Dict` com estado e `action_mask`. O estado separa jogadores,
propriedades, características estáticas do tabuleiro, fase, leilão, transferência hipotecária
pendente, ledger ordenado de dívidas, progresso da insolvência, oferta pendente, regras e estoque de casas/hotéis,
incluindo `house_shortage`. Slots de jogadores são relativos ao observador: slot 0 é o
próprio jogador. As cartas de prisão, propriedades oferecidas e participantes de leilão
também são visíveis; a ordem oculta dos baralhos continua privada.
Quando o agente atual está preso, `jail_forecast` apresenta estimativas do modelo
de movimento e caixa; `jail_decision_forecast` simula opções de saída usando o motor
e políticas de continuação. O contrato detalhado e as hipóteses estão em
[`contrato-observacao-rl-v2.md`](contrato-observacao-rl-v2.md).
O contrato versionado da interface de agente, incluindo ciclo de passos,
recompensa, truncagem, ações estruturadas e reprodutibilidade, está em
[`contrato-agente-rl-v1.md`](contrato-agente-rl-v1.md).

O campo `stationary_economics` acrescenta tráfego estacionário por propriedade e cenários
comparáveis de investimento. `property_arrivals_per_roll` é a taxa de chegada por
lançamento; `utility_dice_total_per_roll` preserva o valor esperado dos dados usados no
aluguel de utilidades e `railroad_rent_multiplier_per_roll` inclui o efeito das cartas de
ferrovia; `expected_rolls_per_turn` converte taxas por lançamento em renda
estimada por turno. As linhas `portfolio_membership` e `portfolio_groups` identificam cada
combinação de imóveis do grupo; `portfolio_development` marca o nível uniforme de
construção (0 = sem construções, 1–4 = casas, 5 = hotel). `portfolio_investment` e
`portfolio_income_per_turn` estão normalizados pelo fator monetário da observação;
`portfolio_gross_yield_per_turn` é a renda bruta estimada por turno dividida pelo
investimento. Renda e rendimento são escalados pelo número atual de adversários ativos.

Essas estimativas são estruturais e de longo prazo, não uma previsão da próxima rodada.
O modelo é por lançamento, tenta duplas na prisão e usa cartas uniformes com reposição;
ignora mudanças futuras de posse, hipotecas, comportamento dos adversários e custos
operacionais. O ROI em um horizonte explícito pode ser calculado separadamente a partir da
renda por turno; ele não presume valor residual de revenda. Para uma compra ou lance, compare
o cenário de posse atual do grupo com o cenário que inclui o imóvel candidato, usando o preço
de compra ou o valor do lance como investimento marginal.

Em `property_decision_valuation`, `active` indica que o observador está decidindo sobre um
imóvel livre, diretamente ou em leilão. `candidate_property` e `owned_group_properties`
identificam o imóvel e a posse atual do observador naquele grupo. A avaliação compara a renda
estacionária bruta do grupo antes e depois da aquisição, ajustada pelos adversários ativos e
por hipotecas existentes; também informa se a compra completa o grupo. Os valores de renda e
`reference_price` são normalizados. `marginal_yield_at_reference_price` é a renda marginal
bruta por turno dividida pelo preço de tabela; `payback_turns` é preço dividido pela renda
marginal por turno (com `payback_available` indicando se essa razão é definida). No leilão,
`next_auction_bid` fornece o menor lance legal atual; o agente pode comparar o retorno
marginal com qualquer lance que esteja considerando. O bloco é somente informativo e não
prescreve uma ação.

`cash_flow_forecast` só fica ativo quando o observador é o jogador que decide comprar um
imóvel. Compara `buy` e `decline` em três horizontes (1, 3 e 5 turnos pessoais), usando 32
simulações pareadas e os mesmos fluxos aleatórios em cada par. Os arrays de cenário usam
ordem `[buy, decline]`; os últimos eixos dos campos de distribuição usam
`[média, p05, p25, p50, p75, p95]`. `cash_change`, `cash_in`, `cash_out`, `ending_cash` e
`minimum_cash` são normalizados pelo fator monetário. As probabilidades de dívida e falência
permanecem entre 0 e 1. Horizonte 1 inclui o restante da vez atual até a próxima vez do
jogador; horizontes 3 e 5 avançam a partir daí.

As trajetórias usam cópias da partida e o motor real para dados, cartas, aluguéis, impostos e
liquidação. Para decisões futuras, todos os jogadores usam `GreedyPolicy`; no cenário de
recusa, o agente focal sai do leilão daquele imóvel. A ordem oculta das cartas é substituída
por embaralhamentos determinísticos dos conteúdos conhecidos, pareados entre os cenários.
Assim, a distribuição é condicional ao estado atual, mas depende dessa política de referência
e não representa uma previsão sem hipótese sobre o comportamento futuro.

`jail_decision_forecast` fica ativo quando o observador está preso e deve escolher entre
`ROLL_DICE`, `PAY_JAIL_FINE` e `USE_JAIL_CARD`, nessa ordem. Para cada opção, simula 1, 3 e
5 turnos pessoais usando o motor e a `GreedyPolicy` nas decisões futuras; informa distribuições
de caixa, probabilidade de continuar preso, dívida e falência, além da quantidade esperada de
cartas de prisão restante. A probabilidade de prisão é o estado no fim de cada horizonte e pode
incluir uma nova ida à cadeia depois da saída inicial.

`property_available` identifica imóveis sem dono no estado atual. Para cada um, os vetores
`property_marginal_income_per_turn` e `property_marginal_yield` mostram sua avaliação
estacionária para o observador, usando a posse atual do grupo e o preço de tabela. O tensor
`unowned_property_probability` estima, por opção e horizonte, a probabilidade de o jogador
encontrar cada imóvel ainda sem dono ao menos uma vez. `weighted_property_opportunity` pondera
essa probabilidade pela renda marginal estacionária do imóvel. É um indicador de oportunidade
de compra, não caixa realizado: a chance de poder pagar e os demais riscos de caixa devem ser
lidos junto com as distribuições de fluxo correspondentes.

Para gestão dos imóveis, `management_decision_valuation` lista por tipo de ação e imóvel
legal o caixa recebido ou gasto imediatamente e a renda estacionária do grupo antes e depois
da alteração. A ordem dos tipos é `BUILD_HOUSE`, `SELL_HOUSE`, `MORTGAGE`, `UNMORTGAGE`.
`marginal_yield_per_turn` é a variação de renda bruta por turno dividida pelo valor absoluto
do caixa movimentado. `management_cash_forecast` fornece, para cada opção legal,
distribuições pareadas entre executar a ação agora e adiá-la uma vez, nos horizontes 1, 3 e
5. Ao adiar, usa uma ação de continuação compatível com a fase (rolar, encerrar a vez ou
encerrar negociações de dívida); depois disso, todos seguem `GreedyPolicy`. As amostras usam
a mesma ordem de estatísticas e escala monetária de `cash_flow_forecast`, com 8 amostras por
comparação. As projeções deixam explícita a hipótese de continuação e não recomendam
automaticamente uma ação.

A ação v2 é um `Dict` com os seguintes campos:

| Campo | Uso |
|---|---|
| `action_type` | Uma das 21 ações listadas abaixo, por índice estável |
| `target_player` | Assento relativo (1 = próximo jogador); `max_players` é o sentinela |
| `tile_id` | ID de casa do tabuleiro; `board_size` é o sentinela |
| `amount` | Lance inteiro do leilão |
| `money_offered`, `money_requested` | Valores inteiros para oferta/contraproposta |
| `offered_properties`, `requested_properties` | Vetores MultiBinary na ordem de `action_map.tile_ids` |
| `offered_jail_cards`, `requested_jail_cards` | Vetores MultiBinary na ordem de `action_map.jail_card_ids` |

Tipos em índices estáveis de 0 a 20: `ROLL_DICE`, `BUY_PROPERTY`, `DECLINE_PROPERTY`,
`END_TURN`, `PAY_JAIL_FINE`, `USE_JAIL_CARD`, `DECLARE_BANKRUPTCY`, `AUCTION_WITHDRAW`,
`AUCTION_BID`, `BUILD_HOUSE`, `SELL_HOUSE`, `MORTGAGE`, `UNMORTGAGE`, `PROPOSE_TRADE`,
`ACCEPT_TRADE`, `REJECT_TRADE`, `COUNTER_TRADE`, `END_DEBT_NEGOTIATIONS`,
`AUCTION_DEBT_PROPERTY`, `KEEP_MORTGAGE`, `REDEEM_MORTGAGE`. As duas ações novas são
acrescentadas ao final, nos índices 19 e 20; a ordem é mantida em `ACTION_TYPES`.

`StructuredActionSpace.action_mask()` devolve máscaras para o tipo, destinatário,
imóvel e itens de troca, além de limites inclusivos de dinheiro. Para ações condicionais,
uma política autoregressiva pode pedir máscaras sucessivas passando `action_type`,
`target_player_index`, `money_offered` e `money_requested`. Os limites numéricos reduzem o
intervalo de lances e valores de dinheiro sem enumerar cada dólar. A validação final é
sempre feita por `GameLoop.validate()` ao processar a ação, então agentes devem respeitar
as máscaras e descartar ações rejeitadas.

`trade_requirements` na máscara informa as condições conjuntas da proposta: índice 0
indica que a proposta/contraproposta precisa transferir ao menos um item; índice 1 indica
que, durante a insolvência, ela precisa aumentar estritamente o dinheiro líquido do
devedor. Máscaras de propriedades e cartas indicam quais itens podem ser selecionados,
mas não representam cardinalidade nem combinações entre vários vetores. O agente deve
honrar `trade_requirements`; o motor continua validando os termos completos. A conversão
da ação estruturada também exige que o dicionário corresponda ao espaço declarado, sem
coerção de escalares fracionários ou booleanos para inteiros.

Gymnasium v2 retorna o estado e as máscaras dentro da observação; `env.action_masks()`
também pode recalcular a máscara condicional para uma política autoregressiva. PettingZoo
v2 usa o mesmo contrato em cada turno AEC; a máscara condicional pode ser recalculada por
`env.unwrapped.action_mask(agent, ...)`. O espaço estruturado requer uma política/modelo
capaz de produzir campos condicionais; não é compatível diretamente com políticas que só
emitem o índice discreto da v1. O `check_env` oficial do Gymnasium amostra ações sem usar
essa máscara; para esse diagnóstico, crie o ambiente com `on_illegal="terminate"`. Em uso
normal, `on_illegal="raise"` ajuda a detectar políticas que enviam ações inválidas.
