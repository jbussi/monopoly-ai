# Treinamento PPO em liga

O treino atual usa PPO categórico mascarado em partidas completas de quatro
jogadores. O agente neural controla um assento e escolhe entre as ações legais
produzidas pelo motor. Seus três oponentes iniciais são as políticas `random`,
`aggressive` e `conservative`, implementadas em
[`app/envs/baselines.py`](../app/envs/baselines.py).

## Políticas de referência

- **Random** sorteia uniformemente entre as ações legais.
- **Aggressive** prioriza compras, lances, construções e resgates de hipoteca;
  aceita trocas recebidas.
- **Conservative** preserva uma reserva de caixa, limita compras e lances e
  rejeita trocas recebidas.

As três consultam as ações fornecidas por `GameLoop.legal_actions()`, então a
heurística escolhe entre movimentos aceitos pelo motor. Elas são adversários de
treino simples e reprodutíveis, não agentes econômicos otimizados.

## PPO e banco de políticas

Cada episódio coleta decisões do agente neural até a partida terminar ou atingir
o limite de turnos. PPO atualiza os pesos em lotes de episódios. Ao fim de cada
geração, o treinador congela a rede em `runs/policy_bank/generation_XXXXX.pt` e
atualiza `manifest.json`, incluindo a quantidade de partidas contra cada perfil.
O número de jogadores é configurável de 2 a 12: um assento usa a rede PPO e os
demais recebem uma política selecionada pela liga.

O ator compartilha os pesos para todas as ações candidatas. Ele combina um
embedding aprendido do tipo da ação, um encoder dos campos da candidata e um
encoder das features condicionais de decisão. Para trocas, essas features
incluem cenários, efeitos e forecasts dos dois participantes e o perfil
comportamental. O estado global é codificado em separado; a máscara restringe
a escolha às ações legais. O crítico estima o valor do estado e não escolhe
ações.

No começo, toda partida enfrenta as três heurísticas. Quando já há políticas
arquivadas, a probabilidade configurada substitui um assento por uma política
histórica sorteada. Assim, o agente enfrenta uma população crescente em vez de
se ajustar somente a um oponente atual. Ao iniciar outra execução com o mesmo
diretório, as políticas do banco são carregadas para compor essa população. A
rede treinável começa com pesos aleatórios; o banco serve como conjunto de
oponentes, não como inicialização dos pesos.

O objetivo terminal continua sendo +1 por vitória, −1 por derrota e 0 em
truncagem. Não se atribui valor artificial ao patrimônio em partidas truncadas.
Este ciclo reduz a exposição a um adversário fixo, mas não garante ausência de
overfitting: os resultados ainda precisam ser medidos em seeds reservadas e
contra perfis fora do conjunto de treino.

## Executar

Instale as dependências opcionais e abra
[`notebooks/treinamento_liga_ppo.ipynb`](../notebooks/treinamento_liga_ppo.ipynb).
O notebook está configurado como piloto de 100 partidas de quatro jogadores.
Aumente `GAMES_PER_GENERATION` e `NUM_GENERATIONS` para um treino longo e
acompanhe duração, truncagens e progresso por partida. O custo depende bastante
dos turnos necessários e dos cálculos de analytics em cada observação.

## Forecasts e tempos por etapa

Os forecasts analíticos de caixa e as features de decisão usam agora 1 e 2
rodadas futuras. O modelo de movimento Markov é mantido num cache limitado por
configuração e regras, e seus
resultados exatos de dados/cartas por estado são reutilizados entre decisões e
cenários clonados. Nas decisões de gestão, o caminho de caixa do estado-base é
calculado uma vez e compartilhado entre as ações candidatas. As rendas e yields
estacionários de portfólio continuam disponíveis, e as projeções de cenário
usam os horizontes curtos de 1 e 2 rodadas.

Os registros de cada episódio incluem tempos de política, avanço do ambiente,
codificação da observação/candidatas e análise de trocas. Também incluem os
tempos e quantidades de chamadas dos caminhos de caixa condicionais, de compra,
gestão e prisão, além do PPO update e do salvamento do checkpoint. Os dados ficam
na linha do episódio retornada pelo treino e no arquivo correspondente em
`runs/policy_bank/episode_status/`. `candidate_encoding_seconds` inclui o tempo
de `trade_analysis_seconds`, e `environment_step_seconds` inclui a observação;
esses campos se sobrepõem e não devem ser somados como durações independentes.
Como o eixo de horizonte da rede mudou, use um novo caminho de checkpoint e
archive para este contrato.

Também é possível configurar diretamente:

```python
from app.training.ppo import PPOTrainingConfig, train_ppo

config = PPOTrainingConfig(
    episodes=3000,
    num_players=4,
    generation_size=100,
    archive_opponent_rate=0.5,
    archive_dir="runs/policy_bank",
    num_workers=4,
    device="auto",
)
agent, history = train_ppo(
    config,
    checkpoint_path="runs/checkpoints/ppo_latest.pt",
)
```

`num_workers` define quantos processos coletam episódios em paralelo. Use `1`
para execução sequencial; valores maiores distribuem partidas pela CPU. Os
episódios de cada lote usam os mesmos pesos e só são atualizados juntos, para
preservar a semântica do PPO. `device="auto"` usa CUDA na otimização da rede
quando PyTorch detecta uma GPU compatível e, caso contrário, usa CPU. A
simulação do jogo continua na CPU; os workers mantêm ambientes próprios e
retornam trajetórias ao processo de treino. Em notebooks no Windows, a criação
de processos usa o método `spawn`, portanto o kernel precisa conseguir importar
o pacote do projeto.

Quando `checkpoint_path` é informado, o treinador grava um checkpoint atômico
após cada partida confirmada. Ele contém pesos, otimizador, geradores aleatórios,
geração, histórico e trajetórias que ainda aguardam o próximo update PPO. Para
retomar, passe o mesmo arquivo em `resume_from` e mantenha a configuração do jogo,
PPO e liga; o número de workers e o dispositivo podem mudar. Partidas concluídas
e confirmadas não são repetidas. Uma partida interrompida no meio é reiniciada
desde sua seed determinística, e seu progresso anterior fica registrado.

O diretório `archive_dir/episode_status/` contém um JSON por partida com seed,
oponentes, status, progresso a cada 25 decisões do aprendiz e estado final de
cada assento (caixa, patrimônio, posição, propriedades, falência, casas e
hipotecas). Os status distinguem fila, execução, rollout calculado e checkpoint
confirmado. Se houver apenas um snapshot `generation_XXXXX.pt` de um treino
antigo, ele pode ser usado em `resume_from` como warm start dos pesos; esse
formato antigo não contém o estado do otimizador nem o histórico de partidas.

`TrainingLeague` permite mudar as heurísticas e a frequência de oponentes
históricos. Cada heurística é uma fábrica sem argumentos, por exemplo:

```python
from app.envs.baselines import AggressivePolicy, ConservativePolicy, RandomPolicy
from app.training import PPOTrainingConfig, TrainingLeague, train_ppo

league = TrainingLeague(
    heuristics={
        "random": RandomPolicy,
        "aggressive": AggressivePolicy,
        "conservative": ConservativePolicy,
    },
    archive_opponent_rate=0.5,
)
agent, history = train_ppo(config, league=league)
```

Para uma regra de composição diferente, `CallableTrainingLeague` aceita uma
função `sampler(opponent_count, archive_available, rng, episode_index)` que
retorna exatamente um rótulo por assento. Use rótulos declarados em
`heuristics` ou `"archive"` para sortear uma política histórica. Também é
possível herdar de `TrainingLeague` e sobrescrever `sample_lineup()`.

Os valores de `episodes` e `generation_size` controlam, respectivamente, a
quantidade de partidas e a frequência de salvamento de uma nova política. Os
arquivos em `runs/` são artefatos locais de experimento.

## Limites atuais

- Cada partida tem um jogador PPO e `num_players - 1` oponentes; ainda não é
  treinamento simultâneo de várias redes neurais.
- As políticas históricas são amostradas uniformemente; não há rating, seleção
  por desempenho ou algoritmo PSRO.
- O treinamento começa com pesos novos a cada execução. Políticas antigas entram
  como oponentes, não como continuação dos pesos treináveis.
- A análise condicional de trocas fica ligada por padrão e aumenta o custo de
  decisões de negociação. As heurísticas respondem de forma simples e não fazem
  contrapropostas.
- A arquitetura atual usa embeddings de tipo explícitos. Checkpoints e bancos
  anteriores à versão v2 precisam ser recriados em um novo `archive_dir`.
- O banco arquiva snapshots e exposição a heurísticas, mas ainda não tem
  rotação, poda, comparação automática nem recuperação após geração interrompida.
- Milhares de partidas podem levar muitas horas. Faça um piloto curto antes de
  reservar um treino longo e use seeds separados na avaliação.
