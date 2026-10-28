# Escopo e preparação da release v1

Este documento descreve o escopo funcional da primeira versão do Monopoly AI. `monopoly_v1`
é o identificador do contrato dos ambientes de RL; a versão do pacote Python continua em
`0.1.0` no `pyproject.toml`. Esta documentação prepara o projeto para uma release, mas não
publica nem cria uma tag.

## Incluído na v1

- Motor de partidas configurável por JSON, com `GameState`, `GameLoop`, ações legais e
  reprodutibilidade por seed.
- Variante `classic_us`, com compra, leilão, aluguel, impostos, prisão, cartas, construção,
  hipotecas, resolução de dívida e falência conforme as simplificações descritas em
  [ambiente.md](ambiente.md).
- Partidas da engine com 2 a 12 participantes. `settings.json` é a fonte oficial para
  valores e flags das regras; `board.json` descreve o tabuleiro e os aluguéis.
- Interfaces Gymnasium para agente único contra oponentes fixos e PettingZoo AEC para
  partidas multiagente. Os ambientes v1 aceitam de 2 a 4 participantes, pois a observação
  tem quatro assentos.
- Espaço discreto de ações, máscara de legalidade, observação numérica, recompensa de
  vitória/derrota, renderização textual e baselines aleatório e guloso.

## Fora do escopo da v1

- API HTTP, persistência de partidas e gestão de sessões.
- Agentes treinados, scripts de treinamento, checkpoints e rastreamento de experimentos.
- Trocas entre jogadores e outras regras econômicas ampliadas.
- Regras avançadas de dívida, incluindo tratar de forma exata a disponibilidade de casas
  ao liquidar hotéis. As regras atuais de dívida são uma implementação simplificada.
- Jackpot de estacionamento livre, timeout de leilão, juros sobre imóveis hipotecados
  herdados e leilão de imóveis devolvidos ao banco.
- Ambientes RL para mais de quatro participantes.

## Limitações conhecidas

O cálculo de liquidação considera o valor de venda de hotéis sem verificar o estoque de
casas do banco. A engine, porém, impede a venda de hotel se não houver quatro casas
disponíveis. Em estados raros com dívida e essa falta de estoque, o cálculo pode indicar
recursos suficientes enquanto `legal_actions()` retorna uma lista vazia. Isso pode travar
uma partida ou um oponente automático. A revisão da dívida deve resolver essa divergência;
até lá, a v1 não deve prometer que todo estado de dívida é recuperável.

O Gymnasium também espera que o agente respeite a máscara de ações. Entradas inválidas
fora do intervalo do espaço discreto não têm tratamento uniforme em todos os modos do
adaptador.

## Instalação e verificação

Requisitos: Python 3.11 ou superior. A engine usa NumPy; Gymnasium e PettingZoo são
dependências opcionais para os ambientes RL.

```bash
pip install -e ".[dev]"
pytest -q
```

Na verificação documentada para esta preparação, a suíte completa passou: **51 testes
aprovados, nenhum ignorado**, incluindo as verificações de API do Gymnasium e PettingZoo,
com Gymnasium 1.4.0 e PettingZoo 1.27.0. Os testes incluem partidas aleatórias de 2 a 4
jogadores, uma partida aleatória de 12 jogadores limitada a 120 turnos, invariantes,
reprodutibilidade, clonagem, máscaras e checadores das interfaces.

## Evolução planejada para v2

A v2 poderá ampliar as regras de dívida, adicionar trocas entre jogadores e implementar
agentes treináveis e seus experimentos de avaliação. Mudanças no espaço de ações ou na
observação devem usar uma nova versão do contrato; manter a v1 disponível evita invalidar
modelos existentes.

## Checklist antes de publicar

- Revisar e aceitar ou corrigir a limitação de dívida descrita acima.
- Definir a versão de distribuição do pacote: o contrato é `monopoly_v1`, mas o pacote
  ainda está marcado como `0.1.0`.
- Rodar `pytest -q` no ambiente de release e registrar as versões de Python e dependências.
- Criar a tag e publicar o pacote somente quando o responsável pela release autorizar.
