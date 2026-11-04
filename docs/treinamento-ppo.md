# Treinamento inicial com PPO

O primeiro treinador do projeto é uma implementação experimental de PPO
categórico mascarado em [`app/training/ppo.py`](../app/training/ppo.py). Ele
treina um agente Gymnasium contra uma mistura reproduzível de `RandomPolicy` e
`GreedyPolicy` com reservas de caixa diferentes.

## Como funciona

O ator recebe a observação e as ações candidatas legais. Uma rede neural codifica
o estado, pontua cada candidata por suas características e amostra uma ação da
distribuição categórica. A máscara impede escolher candidatas indisponíveis. O
crítico estima o retorno esperado do estado; PPO usa essa estimativa para
atualizar a política sem mudanças excessivamente grandes por lote.

A recompensa continua sendo a padrão do ambiente: +1 pela vitória, −1 pela
falência e 0 nos demais casos. O treinador usa `gamma=1.0`. Se a partida é
truncada, a recompensa continua zero e o valor do crítico no estado-limite é
usado como bootstrap; isso não converte caixa ou patrimônio em recompensa.

## Começar

Instale as dependências opcionais no ambiente escolhido:

```bash
pip install -e ".[notebooks]"
```

Execute o notebook
[`notebooks/avaliacao_e_treinamento_ppo.ipynb`](../notebooks/avaliacao_e_treinamento_ppo.ipynb).
Ele compara baselines, salva CSVs e cria gráficos. O bloco PPO fica desligado por
padrão; ligue `RUN_PPO` depois de escolher o tamanho do experimento. O notebook
e o treinador podem ser usados diretamente:

```python
from app.training.ppo import PPOTrainingConfig, train_ppo

config = PPOTrainingConfig(
    episodes=100,
    seed=2026,
    num_players=2,
    max_turns=1000,
    episodes_per_update=8,
    trade_analysis_mode="disabled",
)
agent, history = train_ppo(
    config,
    checkpoint_path="runs/checkpoints/ppo_candidate.pt",
)
```

Os resultados por episódio e as avaliações periódicas são retornados em
`history`. `progress_callback`, quando informado, recebe eventos de início e
fim de cada episódio, além das métricas disponíveis naquele momento. O
checkpoint inclui pesos, estado do otimizador e configuração. As saídas em
`runs/` ficam fora do controle de versão.

Para avaliação final ou comparações reproduzíveis, use
`load_ppo_checkpoint()` e `evaluate_ppo_matchups()`. A segunda função executa
cada baseline em separado, reutiliza seeds entre confrontos e alterna os
assentos de forma equilibrada. Retorna uma linha por partida para cálculo de
intervalos e auditoria; o notebook constrói os resumos e gráficos a partir
desses dados.

## Limites desta primeira implementação

- O espaço de ação é limitado às candidatas geradas por `LegalCandidateProvider`;
  uma ação que não é gerada não pode ser selecionada pela rede.
- `trade_analysis_mode="disabled"` é o padrão inicial. O fluxo de troca tem
  suporte a features no ambiente candidato, mas ainda precisa de validação de
  custo e qualidade antes de ser ligado durante treinamento extenso.
- As baselines existentes rejeitam ofertas recebidas. Por isso, este primeiro
  grupo de oponentes não fornece experiência útil para aprender aceitação ou
  contrapropostas; a negociação exige oponentes próprios capazes de negociar.
- A rede e o treinador não têm ajuste automático de hiperparâmetros,
  gerenciamento de experimentos ou retomada de treino pelo checkpoint.
- A avaliação integrada usa seeds distintas e uma mistura de oponentes. A
  avaliação formal por confronto, com seeds reservadas e intervalos, deve usar
  o protocolo de [`contrato-avaliacao-agentes.md`](contrato-avaliacao-agentes.md).
- A geração das observações inclui analytics e é custosa. Meça a duração de uma
  partida curta antes de escolher um orçamento grande de episódios.

Este código serve para validar o primeiro ciclo de coleta de partidas,
atualização PPO e análise gráfica. Resultados curtos são apenas smoke runs, não
evidência de que a política aprendeu a jogar bem.
