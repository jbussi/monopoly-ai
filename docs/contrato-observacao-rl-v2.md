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
do estado. A previsão inicial está em `jail_forecast` e cobre 1, 3 e 5 turnos
próprios, comparando tentar duplas, pagar a multa e usar uma carta. O número de
amostras inicial é 64; as mesmas sementes por cenário permitem comparações mais
estáveis. O padrão das linhas de `jail_forecast` é `try_doubles`, `pay_fine`, `use_jail_card`;
as colunas de horizonte seguem a ordem 1, 3 e 5.

`jail_forecast` inclui variação e média de caixa, dívida, probabilidade de ainda
estar preso, aluguéis e impostos cobrados, salário de saída, oportunidades de
compra por imóvel, cartas de prisão restantes e distribuição da posição final.
Valores monetários seguem a escala da observação. A previsão não soma esses
resultados em um escore nem recomenda uma ação.

Limites desta primeira versão: após a decisão inicial, um jogador que voltar à
prisão tenta duplas nos turnos seguintes; imóveis livres são recusados e os
leilões ficam sem lances; os outros jogadores não avançam nem tomam decisões;
uma dívida não resolvida interrompe a trajetória. As posições e ativos atuais
dos oponentes permanecem no cálculo dos efeitos de aluguel. Cartas são
amostradas a partir da composição pública, sem usar a ordem escondida do motor.
Essas hipóteses serão refinadas com o tempo. As estimativas descrevem trajetórias
condicionais, não o lucro final esperado da partida.
