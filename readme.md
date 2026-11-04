# Monopoly AI

Motor de Banco Imobiliário configurável, análises econômicas e ambientes para
pesquisa com agentes. O motor e os adaptadores RL estruturados aceitam de 2 a 12
jogadores; o ambiente legado v1 mantém sua observação fixa para até quatro.

## Comece por aqui

1. [Arquitetura e mapa dos pacotes](docs/arquitetura.md)
2. [Ambientes, regras e contratos](docs/ambiente.md)
3. [Contrato de agente RL](docs/contrato-agente-rl-v1.md) e
   [contrato de observação v2](docs/contrato-observacao-rl-v2.md)
4. [Informações condicionais para decisão](docs/informacoes-de-decisao.md)
5. [Estado dos agentes e treinamento](agents/README.md)

## Pacotes principais

```text
app/
├── core/                  configuração geral do processo
├── application/           montagem de partidas a partir de uma variante
├── infraestructure/       leitura, modelos e validação dos JSONs
├── domain/
│   ├── game/              estado, ações e regras do motor
│   ├── analytics/         análises de tabuleiro, economia e decisões
│   └── agents/            protocolos e adaptadores de políticas
├── envs/                  Gymnasium, PettingZoo, baselines e simulação
├── api/                   dependências preparatórias; sem API HTTP implementada
└── main.py                ponto de entrada da aplicação
configs/games/classic_us/  configuração da variante clássica
data/analytics/            notebooks e artefatos exploratórios
docs/                      contratos e guias do projeto
tests/                     regras, análises e adaptadores
```

As responsabilidades e o fluxo entre pacotes estão em
[docs/arquitetura.md](docs/arquitetura.md). A pasta `agents/` contém documentação;
os agentes executáveis atuais são as baselines e os adaptadores descritos no guia.

## Instalação

Requer Python 3.11 ou superior.

```bash
pip install -e ".[dev]"
```

O pacote principal requer NumPy. O extra `envs` instala Gymnasium e PettingZoo;
o extra `dev` acrescenta pytest.

## Uso rápido

Crie uma partida e consulte ações legais antes de aplicá-las:

```python
from app.application.game_setup import create_game

loop = create_game(["Ana", "Bia"], seed=2026)
action = loop.legal_actions()[0]
result = loop.process_action(action)
```

Simule baselines:

```bash
python -m app.envs.simulate --games 100 --policies greedy random --seed 2026
```

Para ambientes v2, instale o extra `envs` e siga os contratos de ambiente e de
agente. O projeto ainda não inclui uma API HTTP nem um algoritmo ou pipeline de
treinamento.

## Verificações

```bash
pytest -q
```

O checker oficial do PettingZoo pode levar vários minutos, pois exercita
episódios completos com sequências aleatórias de ações. Consulte
[docs/ambiente.md](docs/ambiente.md) para os contratos e suas limitações.
