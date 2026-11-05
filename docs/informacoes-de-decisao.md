# Contrato de informações para decisão

## Fluxo para uma política

1. Leia a observação do ambiente e sua `action_mask` para conhecer o estado e as ações legais.
2. Monte uma ação candidata. Para ações com parâmetros, como lance, propriedade ou oferta de troca, use os campos estruturados do espaço de ação e suas máscaras condicionais.
3. Quando a ação depender de comparação, chame `evaluate_candidate` com a candidata e uma alternativa legal (`baseline_action`). Reutilize o mesmo `seed` ao comparar candidatas.
4. Consuma os dados retornados sem tratar nenhum campo como recomendação ou recompensa implícita.

Gymnasium expõe essa operação em `env.evaluate_candidate(action, ...)`; PettingZoo expõe `env.evaluate_candidate(agent, action, ...)`. O retorno Python é um `DecisionContext`. `context.to_dict()` ou `decision_context_to_dict(context)` produz um payload serializável, com `schema_version` para identificar o contrato. Para obter o contexto completo e as features de modelo em uma chamada, use `evaluate_candidate_features`; ela retorna um `EncodedDecisionContext` com os campos `context` e `features`.

## Seções do retorno

Um `DecisionContext` sempre identifica `player_id`, fase, ação candidata, alternativa, seed e número de simulações. As seções específicas permanecem explícitas; as que não se aplicam à ação retornam `null` no dicionário.

| Seção | Ações e conteúdo |
|---|---|
| `candidate` e `baseline` | Cenários comuns: mudança imediata de caixa e patrimônio contábil, avaliação de portfólio e distribuições de fluxo de caixa. Exigem ação alternativa legal. |
| `jail_analysis` | Pagar, usar carta ou tentar sair rolando; oportunidades condicionais de compra e previsões por horizonte. |
| `auction_analysis` | Lances candidatos, chance de vitória, preço final simulado, comportamento observado dos oponentes e efeitos para os participantes. |
| `trade_analysis` | Efeitos condicionados a aceitar/recusar ou enviar proposta, avaliação de ambos os lados e histórico de comportamento. |
| `debt_analysis` | Cenários das ações legais de resolução, com efeitos imediatos e forecasts para devedor e credores. Ofertas estruturadas adicionais podem ser passadas em `debt_action_candidates`. |

Nas distribuições, cada horizonte inclui `horizon_rounds`; use esse identificador para localizar o resultado, não a posição do item na lista. Históricos e listas de candidatos também são arrays de tamanho variável.

## Ações que não pedem comparação própria

`ROLL_DICE` fora da prisão apenas inicia o fluxo da jogada. `END_TURN` conclui as decisões disponíveis naquele turno. A observação atual, a fase e a máscara informam quando essas ações são legais. Na prisão, `ROLL_DICE` faz parte de `jail_analysis`, pois concorre com pagar multa ou usar uma carta.

## Reprodutibilidade e hipótese do forecast

Os forecasts de caixa usam como padrão os horizontes de 1 e 2 rodadas e as probabilidades do
modelo de Markov, condicionadas às posições atuais, ao estado da prisão, às cartas configuradas
e à carteira atual. Após aplicar a ação
candidata, propriedades, construções e hipotecas permanecem fixas; nenhuma política escolhe
ações futuras. O movimento e os efeitos monetários geram momentos sob o modelo de Markov; os
percentis são uma aproximação normal. Cartas são amostradas com reposição, e o aluguel de
utilidade usa a média do lançamento adicional dos dados. Por isso, `seed` e `simulations` não
alteram o forecast de caixa, embora continuem relevantes para análises amostradas como leilões.
O código de análise não escolhe a melhor ação nem pondera retorno e risco; essa interpretação
cabe à política.

## Consumo por modelos de RL

O dicionário serializado preserva a estrutura completa dos cenários e valores opcionais. Para modelos que precisam de tensores de tamanho fixo, o projeto fornece o codificador numérico abaixo. Arquiteturas com requisitos diferentes podem criar outro codificador, mantendo IDs explícitos de ação, propriedade, jogador e horizonte.

O projeto também fornece `DecisionFeaturesEncoder(loop, ...)` para essa entrada numérica. Configure `max_alternatives` para a quantidade máxima que o modelo aceita e `horizons` para os horizontes que ele foi treinado para receber. `encoder.encode(context)` retorna arrays `float32`; `encoder.observation_space` descreve suas formas. `alternative_mask` identifica linhas preenchidas. O encoder falha explicitamente se houver alternativas demais ou um horizonte não configurado, em vez de truncar dados silenciosamente.

Os eixos são: alternativa, jogador relativo ao agente, horizonte e imóvel na ordem dos imóveis do tabuleiro. Os arrays principais são:

| Array | Conteúdo |
|---|---|
| `actions` | one-hot do tipo de ação, valores e parâmetros estruturados, incluindo bitmasks de imóveis e cartas oferecidos/solicitados. |
| `scenario_metrics` | estatísticas da simulação do leilão por lance candidato, incluindo preço final e custos de hipoteca. |
| `player_effects` | caixa e patrimônio imediatos; para leilões também probabilidade de vitória, posse do imóvel e caixa recebido. |
| `cash_forecasts` | média, desvio, percentis, caixa final e probabilidades de falência/fim de jogo. |
| `portfolio_values` | renda atual/projetada, fluxo operacional, fluxo líquido, ROI e VPL, com máscaras para valores ausentes. |
| `property_opportunities` | oportunidades de compra durante previsões de saída da prisão, separadas por horizonte. |
| `auction_property_values` e `redeemed_property_values` | preço de mercado, lance, progresso do grupo e valor contábil, separados entre manter e resgatar a hipoteca. |
| `portfolio_values` e `redeemed_portfolio_values` | avaliação do portfólio com a hipoteca mantida ou resgatada. |
| `opponent_behavior` | estatísticas resumidas de comportamento em trocas e leilões; o histórico completo permanece no `DecisionContext`. |

Os adaptadores aceitam `decision_horizons` e `max_decision_alternatives` no construtor, mantendo o formato fixo entre episódios. `decision_features_space` descreve a saída separada da observação normal; as features são calculadas sob demanda para a ação candidata. Os horizontes solicitados ao analisar uma ação precisam estar configurados no encoder.

Exemplo de uso:

```python
encoded = env.evaluate_candidate_features(
    candidate_action,
    baseline_action=baseline_action,
    seed=42,
)
context = encoded.context       # histórico e detalhes completos
features = encoded.features     # arrays float32 para o modelo
```

Em PettingZoo, passe também o agente decisor: `env.evaluate_candidate_features(agent, candidate_action, ...)`.

## Adaptador de política

`app.domain.agents.FeaturePolicy` conecta um provedor de candidatas, um avaliador que devolve `EncodedDecisionContext` e um seletor (modelo treinado ou scorer de referência). O seletor recebe a partida e a sequência de pares ação/features; a política verifica que sua escolha pertence às candidatas e continua legal no motor. O seletor define a preferência. O adaptador não acrescenta pesos estratégicos.

O provedor padrão usa `loop.legal_actions()` e inclui respostas legais à troca pendente. Propostas de troca precisam de termos estruturados, então devem vir de um provedor personalizado. O avaliador também é injetável para que a estratégia escolha baselines e parâmetros de análise adequados a cada classe de ação.
