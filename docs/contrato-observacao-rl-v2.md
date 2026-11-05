# Contrato de observação RL v2

## Política de informação

Cada observação representa o estado público no momento em que o agente precisa
decidir. Os assentos dos jogadores são relativos ao observador: a posição zero
em campos de jogadores representa o agente que recebe a observação. O contrato
não expõe a identidade interna dos jogadores.

O agente recebe os saldos, posições, situação de prisão e cartas de saída da
prisão de todos os jogadores; propriedade, construções e hipotecas; fase e
decisão pendente; dados já lançados; dívidas, negociações, leilões e transferências
de hipoteca; recursos do banco; características do tabuleiro; e as regras
configuradas para a partida.

## Campos no nível superior

A observação mantém uma forma fixa e agrupa os dados por assunto. Use os
indicadores de disponibilidade dos blocos condicionais; arrays zerados podem
significar que aquela análise não se aplica ao estado atual.

| Campo | Conteúdo |
|---|---|
| `players`, `properties`, `board_features`, `phase`, `pending_tile` | Jogadores em assentos relativos, imóveis, definições do tabuleiro e fase/decisão pendente. |
| `auction`, `debts`, `debt_resolution`, `mortgage_transfer` | Lances, dívidas em ordem, progresso de insolvência e escolha hipotecária pendente. |
| `pending_trade`, `trade_history` | Termos da oferta ativa e histórico público de ofertas e respostas. |
| `globals`, `dice` | Estoque do banco, turno e dados configurados/lançados. |
| `rules`, `rules_complete` | Regras legadas e parâmetros nomeados completos. Prefira `rules_complete`. |
| `deck_card_counts` | Composição agregada das pilhas, sem revelar a ordem futura. |
| `jail_forecast`, `jail_decision_forecast` | Estimativas do modelo de movimento e previsões condicionais das opções da prisão. |
| `stationary_economics`, `property_decision_valuation` | Frequências estacionárias e valuation estrutural da aquisição de imóveis. |
| `cash_flow_forecast`, `management_decision_valuation`, `management_cash_forecast` | Fluxos de caixa e avaliação das decisões de compra, construção, venda e hipoteca. |
| `action_mask` | Tipos, opções, limites numéricos e condições de legalidade da ação atual. |

Para definições de unidades, eixos e premissas das avaliações econômicas, veja
[contratos de ambiente e regras](ambiente.md).

`board_features` usa uma linha por casa na ordem do tabuleiro. Suas colunas
iniciais mantêm o layout já existente para tipo de casa, valores e aluguéis;
depois vêm os indicadores one-hot dos grupos de propriedades e dos tipos de
casa especial, e por último a opção percentual de cada casa de imposto dividida
por 100. Assim, casas especiais diferentes e impostos percentuais não são
colapsados na mesma representação genérica. As colunas de grupo e casa especial
seguem ordem lexicográfica dos rótulos presentes na configuração do tabuleiro.

## Baralhos

`deck_card_counts[baralho][carta]` é um vetor com duas contagens, nesta ordem:

1. cartas atualmente na pilha de compra;
2. cartas na pilha de descarte.

Os IDs das cartas identificam as posições do campo; os valores revelam somente
a composição agregada das pilhas. A observação não contém a ordem embaralhada,
a próxima carta, o estado do gerador aleatório ou resultados futuros. Cartas
“saia da prisão” mantidas por jogadores aparecem em `players.jail_cards` e não
são contadas nas pilhas até serem devolvidas.

## Regras

`rules_complete` contém os parâmetros nomeados que alteram decisões ou limites
do jogo. Valores monetários são divididos por `money_scale`; posições do
tabuleiro são vetores one-hot. `max_turns_enabled` distingue limite ausente de
limite ativo, e `max_turns` é normalizado pelo limite configurado no ambiente.

O campo legado `rules` permanece por compatibilidade com consumidores v2
existentes. Novos consumidores devem preferir `rules_complete`, cujos nomes e
semântica são estáveis e explícitos.

## Reprodutibilidade e ausência de vazamento

A seed continua sendo parâmetro de inicialização do ambiente para reproduzir
partidas, não uma observação do agente. Para estados públicos iguais, mudar
somente a ordem interna futura das cartas não deve alterar a observação.

## Informações por decisão já mapeada

| Decisão | Campos observáveis usados |
|---|---|
| `ROLL_DICE` | `phase`, `players.current_actor` e `action_mask.action_type`. Se a máscara autorizar a ação, não há escolha de resultado nem analytics adicional. |
| `END_TURN` | `phase`, `players.current_actor`, máscara e estado atual. O agente encerra depois de avaliar as demais ações legais da vez; ações já executadas estão refletidas no estado. |
| Sair da prisão por multa, carta ou tentativa de dupla | Dinheiro, estado e tentativas na prisão, cartas em posse, posições e atividade dos jogadores, imóveis livres/ocupados, progresso da partida, regras de prisão e ações legais. `dice.double_probability` dá a chance exata de dupla em um lançamento dos dados configurados. |

As propriedades livres podem ser identificadas em `properties.owner`: a primeira
coluna representa imóveis sem proprietário. A observação também entrega as
características econômicas dessas casas em `board_features` e a composição
pública dos baralhos em `deck_card_counts`.

O retorno esperado de voltar a circular é uma estimativa derivada, não um fato
do estado. `jail_forecast` compara tentar duplas, pagar a multa e usar uma carta,
nessa ordem, nos horizontes próprios de 1, 3 e 5 turnos. `jail_decision_forecast`
usa 1 e 2 rodadas para as probabilidades de movimento e fluxo monetário com a
carteira fixa; consulte
`horizons`, `option_available` e `sample_count`. Nesse forecast analítico,
`sample_count` é zero, pois não são coletadas amostras.

`jail_forecast` informa métricas do movimento e da oportunidade de circular,
incluindo posição, caixa, dívidas, aluguel/impostos cobrados e imóveis livres
encontrados. `jail_decision_forecast` acrescenta distribuição aproximada de caixa,
probabilidade de saldo negativo, chance de ainda estar preso, cartas restantes e
oportunidades condicionais por propriedade. As hipóteses de carteira fixa e
aproximação normal estão descritas no [guia de ambiente](ambiente.md). O forecast
descreve cenários condicionais e não recomenda uma ação nem estima o lucro final
da partida.
