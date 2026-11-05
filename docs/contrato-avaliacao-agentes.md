# Contrato de avaliação de agentes

Este documento define como comparar políticas e checkpoints de aprendizado no
Monopoly AI. A avaliação mede desempenho no jogo; não altera as recompensas do
ambiente nem pontua o estado de uma partida truncada.

## 1. Resultado de uma partida

Cada partida termina em uma destas categorias:

- **Vitória:** o motor declarou um vencedor.
- **Empate:** a partida terminou sem vencedor e sem truncagem.
- **Inconclusiva:** atingiu `max_turns`; não recebe recompensa terminal especial.

Uma partida inconclusiva conta como não vitória no cálculo de vitórias por
partida. Também é reportada separadamente. A taxa de vitória entre partidas
decididas é uma métrica secundária e sempre aparece junto da taxa de
inconclusivas, para não esconder partidas que chegaram ao limite.

## 2. Métrica principal

A métrica principal é a **taxa de vitória por partida**:

```text
vitórias do agente / total de partidas disputadas
```

Partidas inconclusivas permanecem no denominador e não contam como vitória. Em
partidas com mais de dois jogadores, cada jogo conta uma vez para cada política
ou grupo avaliado, não uma vez por assento. Também se informa a taxa de vitórias
por assento para revelar desequilíbrio de posição.

Toda taxa de vitória deve vir acompanhada de intervalo de confiança de 95%,
quantidade de partidas e quantidade de inconclusivas. Ao comparar duas versões
nos mesmos jogos, seeds e assentos, a diferença entre versões deve ser calculada
de forma pareada; intervalos independentes não representam essa dependência.

## 3. Métricas secundárias

Reportar junto da métrica principal:

- taxa de partidas decididas e taxa de inconclusivas;
- vitórias por assento e, em partidas multi-jogador, posição final quando
  disponível;
- falências, turnos/rodadas e decisões médias;
- caixa, patrimônio e imóveis médios ao fim da partida;
- ações ilegais — qualquer ação ilegal é uma falha de compatibilidade;
- para análises específicas de troca: propostas, aceitações, recusas e
  contrapropostas, sem tratá-las como recompensa.

Caixa, patrimônio e imóveis são diagnósticos. Não substituem a métrica de
vitória para escolher o melhor agente.

## 4. Conjunto de oponentes e cenários

Manter políticas e parâmetros fixos durante uma rodada de avaliação. O conjunto
inicial de referência é:

- `random`, como teste de sanidade;
- `aggressive` e `conservative`, que compõem a liga inicial do treinamento PPO;
- `greedy:0`, `greedy:25`, `greedy:50` e `greedy:150`, para cobrir diferentes
  reservas de caixa da baseline Greedy.

Não condensar os resultados desses oponentes em uma única média sem também
mostrar cada confronto. Para a comparação principal, executar partidas de dois
jogadores contra cada baseline. Como teste multi-jogador, usar quatro jogadores
com o agente avaliado contra três cópias da mesma baseline. Avaliações com 6 e
12 jogadores são testes de robustez adicionais e devem ser apresentadas por
separado.

Os resultados de partidas com números diferentes de jogadores não são
agregados: cada tamanho é um cenário distinto.

## 5. Seeds, assentos e reprodutibilidade

- Usar a mesma lista de seeds ao comparar agentes, checkpoints e baselines.
- Rotacionar o agente avaliado por todos os assentos possíveis de forma
  equilibrada; para grupos, equilibrar também a distribuição dos assentos do
  grupo.
- Registrar versão do código/contrato, configuração de regras, número de
  jogadores, limite de turnos, políticas e parâmetros dos oponentes, seeds e
  checkpoint avaliado.
- Guardar os resultados por partida em CSV para permitir auditoria e análises
  pareadas.
- Separar seeds de treino, validação e teste final. Usar validação para escolher
  checkpoint; reservar o conjunto de teste para a comparação final.

O simulador já deriva seeds consecutivas a partir de `--seed`, rotaciona os
assentos e pode salvar dados individuais por partida com `--csv`. Portanto, uma
mesma configuração de avaliação é reproduzível.

O tamanho inicial das rodadas de avaliação fica definido assim:

- **Validação de checkpoints:** 100 partidas por confronto de dois jogadores e
  100 partidas por baseline no cenário de quatro jogadores. Para quatro
  jogadores, 100 permite distribuir 25 partidas para cada assento.
- **Teste final:** 500 partidas por confronto de dois jogadores e 200 partidas
  por baseline no cenário de quatro jogadores. Esses totais permitem rotação
  uniforme dos assentos e dão mais precisão ao resultado final.
- **Robustez com 6 ou 12 jogadores:** 120 partidas por baseline e composição de
  assentos, distribuídas igualmente entre assentos. Não substituem o teste
  principal de dois e quatro jogadores.

Se a comparação ficar inconclusiva estatisticamente ou muito próxima, ampliar
as partidas pareadas com novas seeds antes de concluir que houve melhora.

## 6. Limite de duração

O limite de turnos deve ser idêntico em todos os agentes de um confronto e
constar no relatório. A recompensa padrão permanece `+1` para vitória, `-1`
para falência e `0` nos demais casos, inclusive ao truncar. Nenhum valor de
caixa, patrimônio ou posição é convertido em recompensa terminal na truncagem.

`1000` turnos é o limite inicial considerado para avaliações de dois jogadores,
mas uma truncagem continua sendo inconclusiva, mesmo que o limite pareça
suficiente. O histórico de experimentos do projeto registrou truncagens de
13% em partidas de dois jogadores no limite padrão de 1000 e de 69,5% em um
confronto de quatro jogadores mesmo com limite de 3000. Portanto, o cenário de
quatro jogadores usa inicialmente 3000 turnos; a taxa de truncagem observada
precisa acompanhar todo resultado. Se o limite mudar, repetir o confronto
completo com o novo limite antes de comparar políticas.

## 7. Seleção de checkpoints

Escolher checkpoints somente com o conjunto de validação e pela taxa de vitória
por partida no conjunto fixo de oponentes. Publicar também cada confronto para
evitar que uma melhora contra uma baseline esconda uma regressão contra outra.
Depois de congelar o checkpoint, executar uma única avaliação final no conjunto
de teste reservado. A avaliação final não deve retroalimentar a escolha do
checkpoint.

O protocolo não define um limiar universal de vitória para aprovação. Primeiro
é necessário medir a variabilidade das baselines e da política treinada; as
comparações entre versões devem usar as mesmas partidas pareadas e apresentar
intervalos de confiança.

## 8. Estado da implementação

O simulador em [`app/envs/simulate.py`](../app/envs/simulate.py) implementa
seeds reproduzíveis, rotação de assentos, métricas de vitória, falência,
truncagem, caixa, patrimônio e imóveis, além de exportação CSV. Atualmente,
intervalos de Wilson são calculados para taxas de vitória individuais. O
intervalo pareado da diferença entre agentes, posição final multi-jogador e
métricas específicas de trocas ainda precisam ser adicionados antes de serem
considerados resultados automáticos do protocolo.

## Referências

- [Contrato do agente RL](contrato-agente-rl-v1.md)
- [Simulador de baselines](../app/envs/simulate.py)
- [Políticas baseline](../app/envs/baselines.py)
