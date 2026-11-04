# Contrato do agente RL v1.0.0

Este documento especifica a interface entre uma política de aprendizado e os
ambientes `monopoly_v2`. Ele versiona o **contrato do agente**; não altera nem
substitui os nomes dos ambientes `monopoly_v1` e `monopoly_v2`.

O contrato descreve o comportamento implementado nos adaptadores e espaços
atuais. Correções editoriais não alteram a versão. Acréscimos opcionais que não
quebrem consumidores podem receber uma versão secundária. Mudanças que alterem
nomes, formas, ordem de índices, interpretação de máscara, unidade de passo ou
recompensa padrão exigem uma nova versão principal deste contrato. A versão não
está hoje embutida como campo nas observações; consumidores devem fixar a versão
da documentação/dependência.

## 1. Ciclo de decisão

Em cada decisão, a política recebe uma observação do agente decisor, seleciona
uma ação estruturada e a envia ao ambiente. O motor valida a ação completa. A
máscara reduz o conjunto de escolhas possíveis, mas não substitui a validação
final de `GameLoop`.

O observador é sempre o jogador que deve decidir naquele instante. Jogadores e
ativos são representados em posições relativas ao observador, conforme o
contrato de [observação v2](contrato-observacao-rl-v2.md). Informações derivadas
de uma ação hipotética são fornecidas separadamente por
[`evaluate_candidate`](informacoes-de-decisao.md); elas não são uma recomendação
nem uma recompensa implícita.

### Unidade de passo por ambiente

- **Gymnasium v2:** `step(action)` aplica a ação do aprendiz e executa as
  políticas oponentes até o aprendiz voltar a decidir ou a partida terminar.
  Um passo do Gymnasium pode, portanto, conter várias ações do motor.
- **PettingZoo AEC v2:** cada `step(action)` aplica uma ação do agente em
  `agent_selection`. A ordem de decisão é a ordem do motor, inclusive durante
  leilões, negociações, dívidas e transferência de hipoteca.

Não compare diretamente o número de passos dessas interfaces como se fossem a
mesma unidade temporal.

## 2. Entrada: observação

Gymnasium e PettingZoo entregam um `spaces.Dict` com os campos do estado
codificados por `ObservationEncoderV2` e um campo adicional `action_mask`.
Os campos do estado agrupam:

- jogadores, caixa, posição, prisão e cartas de prisão;
- propriedade, dono, construções e hipoteca;
- fase, imóvel pendente, leilão e dívidas;
- estado da liquidação de dívida, oferta e histórico de trocas;
- regras, dados, composição pública dos baralhos e recursos do banco;
- economia estacionária, previsões e avaliações econômicas disponíveis na
  situação atual.

As formas e ordens detalhadas pertencem ao esquema de
[`ObservationEncoderV2`](../app/envs/observation_v2.py) e à especificação de
[observação RL v2](contrato-observacao-rl-v2.md). Dinheiro em observações e
features econômicas segue `money_scale`; taxas e probabilidades permanecem em
seus intervalos documentados. O estado não revela a ordem futura dos baralhos,
nem resultados futuros aleatórios.

`action_mask` tem campos para tipo de ação, alvo, imóvel, ativos negociáveis,
intervalos de valores monetários e requisitos de troca. Máscaras de escolha
discreta são vetores 0/1; `money_bounds` contém limites inteiros inclusivos.
Quando a política seleciona um tipo de ação ou alvo, ela pode pedir ao ambiente
uma máscara condicional mais específica (`action_masks` no Gymnasium;
`action_mask` no PettingZoo sem wrapper, acessado via `unwrapped`).

As máscaras de ativos indicam itens individualmente selecionáveis; elas não
codificam todas as combinações permitidas nem todos os vínculos entre campos.
`trade_requirements` informa condições conjuntas mínimas das ofertas. A política
deve respeitar todos os campos condicionais e o motor ainda pode rejeitar uma
combinação inválida. Ao decidir um passo de ação, use apenas a observação do
agente corrente; máscaras de agentes fora da vez são zeradas.

## 3. Entrada opcional: informação condicionada à ação

`evaluate_candidate` e `evaluate_candidate_features` calculam cenários para uma
ação candidata legal. O agente escolhe quais candidatas analisar; o cálculo pode
ser caro e não ocorre automaticamente para todas as ações possíveis. Os dados
podem incluir forecasts de caixa, valuation de portfólio, consequências de
trocas/leilões/dívidas e perfil observado dos oponentes. As hipóteses, seeds,
formatos e seções opcionais estão em
[`informacoes-de-decisao.md`](informacoes-de-decisao.md).

Essas análises são estimativas condicionais ao estado, às simulações e à
política de continuação configurada (atualmente Greedy nos forecasts de fluxo
de caixa). Não atribuem utilidade, não ranqueiam ações e não substituem os
valores originais do estado.

## 4. Saída: ação estruturada

O agente envia um dicionário com **todos** os campos do espaço, mesmo que alguns
sejam sentinelas ou ignorados pelo tipo de ação:

| Campo | Interpretação |
|---|---|
| `action_type` | Índice em `ACTION_TYPES`, atualmente 21 tipos na ordem definida em [`action_space_v2.py`](../app/envs/action_space_v2.py). |
| `target_player` | Distância do assento do agente na ordem da partida; 0 é o próprio agente. Usado para `PROPOSE_TRADE`; `max_players` é sentinela. O alvo ainda precisa estar ativo e ser autorizado pela máscara. |
| `tile_id` | Posição do tabuleiro. Usado pelas ações de imóvel; `board_size` é sentinela. |
| `amount` | Valor inteiro do lance em `AUCTION_BID`. |
| `money_offered`, `money_requested` | Valores inteiros da oferta/contraproposta. |
| `offered_properties`, `requested_properties` | Vetores binários na ordem `action_map.tile_ids`. |
| `offered_jail_cards`, `requested_jail_cards` | Vetores binários na ordem `action_map.jail_card_ids`. |

As escalas e limites estruturais vêm do espaço de ação; a legalidade econômica
e contextual depende do estado atual. Uma ação só é efetiva se a conversão e a
validação final pelo motor forem bem-sucedidas.

Uma política deve preferir selecionar uma ação candidata legal ou gerar os
campos de forma autoregressiva com máscaras condicionais. Não deve presumir que
qualquer combinação de valores dentro dos limites globais do `spaces.Dict` será
legal. Em Gymnasium, o padrão `on_illegal="raise"` lança erro para uma ação
inválida; `on_illegal="terminate"` encerra o episódio com `illegal_reward`.
PettingZoo v2 lança erro para ação inválida.

## 5. Seeds, reset e memória

`reset(seed=s)` reinicia a partida com uma semente controlada e, no Gymnasium,
reinicia políticas oponentes que implementam `reset(seed)`. Mantidas a mesma
configuração, ordem de assentos e sequência de ações, a seed permite reproduzir
o jogo e a aleatoriedade do ambiente.

A aleatoriedade interna da política aprendiz é responsabilidade do treinador ou
da própria política: ela deve possuir um gerador local e semeá-lo a partir da
seed do episódio fornecida pelo pipeline de treino. O ambiente não inclui a
seed na observação e não distribui uma seed específica ao ator externo. Memória
de oponentes ou memória recorrente deve ser limpa em cada episódio, mantendo-se
apenas quando fizer parte explícita do estado recorrente do agente.

Para comparações de `evaluate_candidate`, use a mesma seed e os mesmos
parâmetros em todas as candidatas do mesmo estado. Isso torna os cenários mais
comparáveis; não elimina a incerteza Monte Carlo.

## 6. Recompensa e encerramento

A recompensa padrão é `win_loss_reward`: +1 para o vencedor, −1 para o jogador
declarado falido e 0 nos demais casos. Ela é esparsa; não recompensa caixa,
patrimônio ou aquisição de imóveis durante a partida. Shaping é uma escolha
explícita do treinador por meio de `reward_fn`.

Fim natural e limite de turnos são sinais distintos:

- vitória/falência encerra o episódio naturalmente (`terminated`);
- atingir `max_turns` encerra por limite (`truncated`), sem vencedor, e a
  recompensa padrão é 0 para todos;
- truncagem não deve ser contabilizada como vitória nem como falência. Relate-a
  separadamente em avaliações e decida explicitamente como o algoritmo de
  treino trata o bootstrap no limite do episódio.

A recompensa padrão de truncagem é neutra por compatibilidade com o
comportamento atual. Uma futura recompensa baseada em patrimônio no limite do jogo será uma
mudança deliberada de objetivo e requer versão/experimento próprios; não é
aplicada implicitamente neste contrato.

## 7. Fronteira dos ambientes e treinamento

- Use **Gymnasium v2** para um aprendiz contra oponentes fixos e transições
  agrupadas até a próxima decisão do aprendiz.
- Use **PettingZoo AEC v2** para múltiplos agentes, self-play e observação de
  cada decisão individual.
- Os dois ambientes aceitam 2 a 12 jogadores. Todos os jogadores usam o mesmo
  espaço estruturado de ação; `max_players` fixa o tamanho das observações e
  deve ser pelo menos `num_players`.
- O contrato não prescreve algoritmo, arquitetura, função de utilidade,
  exploração, modelo de oponente ou shaping. Essas opções precisam acompanhar
  cada experimento para que os resultados sejam interpretáveis.

Um ator Gymnasium pode implementar `RLAgent.act(observation, action_mask,
deterministic=False)`. `run_gymnasium_episode(env, agent, seed=...)` é o rollout
de referência: chama `agent.reset(seed)` quando disponível, reinicia o ambiente
com a mesma seed, passa estado e máscara em cada decisão e devolve retorno,
quantidade de decisões, flags de término/truncagem e indicação de ação ilegal.
O rollout não atualiza pesos nem define um algoritmo de treinamento.

Como alternativa à ação estruturada direta, `CandidateSelectionEnv` envolve o
Gymnasium v2 e apresenta `Discrete(max_candidates + 1)`. Cada linha de
`candidate_actions` codifica uma ação concreta gerada e validada pelo motor;
`candidate_mask` identifica as linhas de jogo disponíveis.
`candidate_action_mask` também pode habilitar um índice extra, `max_candidates`,
para solicitar análise de troca. O ator retorna um índice ativo nesse espaço.
Para esse wrapper, use
`run_gymnasium_episode(..., action_mask_key="candidate_action_mask")`. Essa forma
permite um ator categórico mascarado e mantém o mapeamento do índice para a ação
concreta dentro do ambiente.

No modo padrão `trade_analysis_mode="on_demand"`, selecionar o índice extra
solicita as análises e devolve o mesmo estado com recompensa 0, sem avançar o
jogo. `candidate_decision_features` então contém a codificação fixa do
`DecisionContext` para propostas, respostas e contrapropostas fora da
insolvência; `candidate_analysis_mask` identifica as linhas calculadas. As
features incluem cenários condicionais para os dois participantes e os
horizontes configurados. O padrão limita a quatro candidatas de troca e uma
simulação por análise; `max_trade_candidates`,
`candidate_analysis_simulations` e `candidate_analysis_horizons` podem ser
ajustados no wrapper. `trade_analysis_mode="automatic"` calcula ao observar;
`"disabled"` desativa a consulta. Aceitar e rejeitar uma oferta pendente são
preservadas como opções distintas. Trocas durante a insolvência ainda não
recebem esta análise por candidata; integrar o `DebtResolutionAnalysis` ao
wrapper permanece pendente.

No PettingZoo, `run_pettingzoo_episode(env, agents, seed=...)` recebe um mapa
dos nomes de assento para atores. Em cada turno AEC chama somente o ator
selecionado, soma a recompensa acumulada retornada por `last()` e registra
decisões, término, truncagem e `info` por assento. A seed do ator é derivada da
seed do episódio e do assento. Atores com memória recorrente devem ser instâncias
separadas por assento; se o mesmo objeto for compartilhado, ele é reiniciado uma
vez e o estado interno também será compartilhado entre esses assentos.

## 8. Invariantes para uma implementação de política

Uma política compatível deve:

1. agir somente quando for o agente selecionado;
2. interpretar jogadores, propriedades, cartas e ações na ordem documentada;
3. aplicar máscaras e limites condicionais antes de escolher os parâmetros;
4. emitir um dicionário que pertence ao espaço estruturado;
5. aceitar a validação do motor como autoridade final;
6. manter a aleatoriedade e a memória isoladas por episódio;
7. distinguir vitória, falência, término natural e truncagem;
8. registrar versão do contrato, ambiente, configuração, seeds e políticas
   oponentes em cada avaliação.

No uso direto do motor, a interface comum é estrutural:
`agent(loop: GameLoop) -> GameAction`. `SeedableAgent` acrescenta
`reset(seed: int | None) -> None`. `FeaturePolicy` aplica esse contrato a uma
sequência finita de ações candidatas e verifica a legalidade antes e depois da
seleção. `MaxScoreSelector` é um seletor genérico: recebe uma função de
pontuação externa, escolhe a maior pontuação e mantém a primeira candidata em
caso de empate. Ele não fornece uma função de valor estratégica padrão.

O provedor padrão combina as ações enumeradas pela engine com até 32 candidatas
de troca legais. Ele considera dinheiro, ativos individuais, trocas entre
ativos e amostras reproduzíveis de bundles com até três ativos de cada lado.
Esse conjunto é uma amostra limitada para tornar a análise viável; não contém
todas as combinações de ofertas possíveis. `max_candidates` também limita o
número total de candidatas que a `FeaturePolicy` avalia. Valores em dinheiro são
amostrados entre 0%, 25%, 50%, 75% e 100% do caixa disponível de cada parte;
bundles com 2 ou 3 ativos são amostrados com seed. São candidatas para avaliação,
não uma recomendação de oferta nem uma estimativa de aceitação.

## Referências de implementação

- [Ambiente Gymnasium v2](../app/envs/gymnasium_env_v2.py)
- [Ambiente PettingZoo AEC v2](../app/envs/pettingzoo_env_v2.py)
- [Observação v2](../app/envs/observation_v2.py)
- [Ação estruturada e máscaras](../app/envs/action_space_v2.py)
- [Recompensa padrão](../app/envs/rewards.py)
- [Interface de ator RL](../app/domain/agents/rl_agent.py)
- [Rollout Gymnasium](../app/envs/rollout.py)
- [Rollout PettingZoo AEC](../app/envs/rollout.py)
- [Ambiente de seleção de candidatas](../app/envs/candidate_env.py)
- [Contrato de observação v2](contrato-observacao-rl-v2.md)
- [Informações condicionadas à decisão](informacoes-de-decisao.md)
