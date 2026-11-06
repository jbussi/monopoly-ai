# Treinamento PPO em liga

O treino usa uma política PPO aprendiz por partida, contra heurísticas âncora e
snapshots congelados de políticas anteriores. Uma geração é um bloco de partidas
seguido por um snapshot; não é uma população de redes treinadas em paralelo.
O aprendiz continua seus pesos entre gerações.

## Políticas e rede

As âncoras padrão são `random`, `aggressive` e `conservative`, que escolhem entre
ações aceitas pelo motor. Elas permanecem disponíveis durante toda a execução e
nunca são podadas. A rede compartilhada codifica o estado, o tipo da ação, os
campos da candidata e, quando disponíveis, as features condicionais de decisão.
Uma máscara restringe as escolhas às ações legais. O ator escolhe ações; o crítico
estima o valor do estado.

O PPO, a recompensa e seus hiperparâmetros permanecem independentes da liga. As
partidas concluídas usam a recompensa geométrica simétrica por colocação. Na
truncagem, os eliminados mantêm as recompensas de suas posições e os sobreviventes
empatam e recebem a recompensa de sobrevivência. A definição oficial está em
[`app/envs/rewards.py`](../app/envs/rewards.py); a liga não adiciona shaping.

## Rating multi-jogador

A liga usa **OpenSkill Plackett-Luce**. Ele recebe uma mesa de jogadores com seus
postos e aceita postos iguais para representar empates. Isso se ajusta diretamente
ao resultado do Monopoly: em partidas truncadas, sobreviventes empatam acima dos
eliminados; entre os eliminados, a ordem de falência define os postos. TrueSkill
também suporta free-for-all, incerteza e empates, mas Plackett-Luce foi escolhido
por ser um modelo de ranking multi-jogador direto e recomendado pelo OpenSkill
como equilíbrio entre velocidade e precisão. Veja o
[manual OpenSkill](https://openskill.me/en/v6.0.2/manual.html) e a
[documentação TrueSkill](https://trueskill.org/).

O rating mantém `mu` e `sigma`. A heurística `random` fixa `mu` em `anchor_mu`
para manter a escala; os demais ratings podem variar. O aprendiz recebe dinâmica
contínua de incerteza: antes de cada partida, sua incerteza é aumentada em
quadratura pelo parâmetro `learner_tau`, evitando que o rating congele após muitas
partidas. Cada snapshot herda o rating global e os ratings por número de jogadores,
com `sigma` multiplicado por `snapshot_sigma_inflation`.

O rating usa somente postos. O portão de promoção usa diferença de recompensa.
Esses sinais podem divergir. Ambos são registrados, inclusive por número de
jogadores.

## Admissão e promoção

Cada snapshot passa por duas decisões distintas:

1. **Admissão no arquivo:** compara `mu` do aprendiz antes de inflar `sigma` com o
   maior `mu` do pool. É admitido se não estiver abaixo desse valor mais do que
   `admission_mu_tolerance` permite. Essa regra permissiva evita rejeitar um
   snapshot novo só por ter alta incerteza.
2. **Promoção a campeão:** candidato e campeão jogam mesas pareadas com a mesma
   seed, assento e identidade dos demais oponentes. A diferença é
   `recompensa_candidato - recompensa_campeão`. O portão calcula a média e o
   intervalo por N, depois a média ponderada pela distribuição-alvo. Promove se
   o limite inferior superar `promotion_margin` e se nenhum N com amostra
   suficiente exceder a tolerância de regressão.

As diferenças são calculadas por seed; não se assume que a correlação reduza
muito a variância, pois as partidas podem divergir depois das primeiras decisões.
O intervalo usa o nível configurado e uma correção conservadora para as consultas
sequenciais. O portão joga ao menos `min_gate_games_per_n` e pode parar cedo quando
já há uma decisão; caso contrário, para em `max_gate_games_per_n` por N. Uma
reprovação não interrompe o treinamento do aprendiz.

O portão é **síncrono no limite do snapshot**: enquanto as partidas pareadas
rodam, o aprendiz não coleta novas partidas. Há somente um candidato no portão,
portanto novos candidatos não o substituem nem ficam em fila. Seu resultado
parcial e cada par concluído são persistidos no checkpoint para retomada.

O campeão inicial é `aggressive` por padrão e pode ser escolhido em
`fixed_initial_champion`. Com `None`, a mistura usa as âncoras até o primeiro
snapshot admitido, que vira o campeão inicial sem comparação. Se campeão ou
snapshots recentes não estiverem disponíveis, suas parcelas da mistura retornam
às heurísticas.

## Seleção e sorteio de mesas

`schedule_table()` fica separada do loop de treino e define, nesta ordem: N pela
distribuição configurada, assento do aprendiz, identidades dos oponentes e seed.
N é amostrado de forma estratificada quando habilitado. O assento é uniforme; no
portão, fica fixo por par. O limite de turnos é
`max_turns_per_player × N`.

A mistura sorteia fontes entre campeão, snapshots recentes, arquivo ponderado por
rating e âncoras. A seleção do arquivo usa
`exp(-abs(mu_snapshot - mu_aprendiz) / rating_temperature)`, misturada com um piso
uniforme `rating_probability_floor`. Assim, políticas fracas ou antigas mantêm
chance de aparecer. O mínimo de âncoras e o máximo de snapshots neurais por mesa
são configuráveis.

O ambiente mantém dimensões de observação para 12 assentos, então a rede pode
receber partidas com N variável na mesma execução. A distribuição padrão e o
notebook piloto usam N constante; isso facilita comparações enquanto o custo por
N é medido. O portão é estratificado por N e as métricas/rating específicos por N
ficam no histórico e no manifesto.

## Arquivo de políticas

O manifesto registra geração, rating global e por N, versão da recompensa,
exposição a oponentes, admissão, resultado do portão e condição de campeão. A poda
respeita `max_archive_size`, preservando o campeão, ratings conservadores altos,
políticas recentes e alguns marcos de geração. Snapshots podados deixam de ser
usados como oponentes e seus arquivos são removidos. Âncoras nunca são podadas.

Os manifestos existentes continuam carregáveis: entradas antigas sem rating
recebem um rating inicial neutro, e campos novos são inferidos por defaults. O
checkpoint retomável armazena também ratings, ratings por N, campeão, contagens,
geração, dados PPO pendentes e o estado/resultados do portão.

## Configuração

`LeagueConfig` concentra cadência de snapshot, rating, admissão, portão,
distribuição de N, composição das mesas e poda. Os valores atuais são **defaults
provisórios**, identificados por `defaults_status="provisional"`; devem ser
calibrados com resultados experimentais. O notebook expõe os valores principais
em `LEAGUE_CONFIG` e o MLflow registra o objeto inteiro.

Exemplo de configuração:

```python
from app.training import LeagueConfig, PPOTrainingConfig, train_ppo

league_config = LeagueConfig(
    snapshot_games=100,
    player_count_distribution={4: 1.0},
    max_turns_per_player=250,
    promotion_margin=0.02,
    confidence_level=0.95,
    min_gate_games_per_n=8,
    max_gate_games_per_n=40,
    max_archive_size=32,
)
config = PPOTrainingConfig(
    episodes=3000,
    num_players=4,
    league=league_config,
    archive_dir="runs/policy_bank",
    num_workers=4,
    device="auto",
)
agent, history = train_ppo(config, checkpoint_path="runs/checkpoints/ppo_latest.pt")
```

Para instalar a dependência de rating, use `pip install -e ".[training]"`.
`num_workers` paraleliza a coleta de partidas na CPU; a atualização PPO ocorre no
processo principal e usa CUDA quando disponível. O portão síncrono roda no
processo principal e interrompe temporariamente a coleta.

## MLflow e retomada

Instale `pip install -e ".[notebooks]"` e inicie, na raiz do projeto:

```bash
mlflow server --backend-store-uri sqlite:///runs/mlflow.db --default-artifact-root ./runs/mlflow_artifacts --host 127.0.0.1 --port 5000
```

Execute [`notebooks/treinamento_liga_ppo.ipynb`](../notebooks/treinamento_liga_ppo.ipynb)
e abra `http://127.0.0.1:5000`. O run registra parâmetros, métricas por partida
e por N, métricas do portão, ratings, tempos, gráficos, manifesto, status de
partidas e checkpoints. Os dados locais continuam em `runs/`.

Para retomar, passe o mesmo checkpoint em `resume_from` e mantenha as mesmas
configurações de jogo, PPO e liga. Workers e dispositivo podem mudar. Partidas
confirmadas não se repetem; uma partida interrompida reinicia pela mesma seed. Um
checkpoint antigo sem estado da liga restaura o modelo/otimizador/histórico que
contém e inicia ratings neutros; snapshots antigos do banco são carregados com
rating neutro.

## Limites atuais

- Cada partida tem um aprendiz treinável; os demais são âncoras ou snapshots
  congelados. Não há treino simultâneo de várias redes.
- A promoção mede recompensa terminal, não força uma política a vencer cada
  perfil; mantenha avaliações por baseline e seeds reservadas.
- Os defaults de rating, confiança, margem, mistura e poda ainda são provisórios.
- O portão síncrono pode acrescentar muitas partidas ao fim de cada geração.
- O banco não implementa PSRO, rating por estilo de jogo nem busca de composição
  ótima de mesa.
