# Histórico da release v1.0.0

Este documento registra o escopo publicado como v1.0.0. Ele é um registro
histórico: o projeto evoluiu depois dessa release. Para entender o estado
presente, comece pela [arquitetura](arquitetura.md) e pelo
[contrato atual dos ambientes](ambiente.md).

## Escopo da v1.0.0

A primeira release reuniu o motor configurável de partidas, a variante
`classic_us`, observações e ações para os ambientes de RL e baselines simples.
O motor permitia de 2 a 12 participantes. Os adaptadores então publicados
usavam espaços fixos para até quatro participantes.

Naquele marco, trocas, liquidação expandida de dívidas, agentes treinados e API
HTTP ainda estavam fora do escopo. As regras e limitações documentadas para
essa versão não descrevem necessariamente a implementação atual.

## Evolução posterior

Depois da v1.0.0, o motor recebeu o fluxo de propostas e contrapropostas,
liquidação e leilões de insolvência, além de melhorias para venda de hotéis
quando faltam casas físicas no banco. Os ambientes v2 passaram a expor as
decisões estruturadas e as regras avançadas. Também foram acrescentados
analytics condicionais, forecasts, baselines de simulação e interfaces para
agentes e rollouts.

O projeto ainda não fornece uma API HTTP de partidas nem um algoritmo/pipeline
de treinamento com checkpoints. O plano atual para agentes está descrito em
[`agents/README.md`](../agents/README.md).

## Instalação e testes

As instruções de instalação atuais estão no [README](../readme.md). Execute a
suíte do checkout e registre o ambiente utilizado ao preparar uma nova release;
os resultados numéricos registrados em versões antigas não devem ser tratados
como resultado de testes atuais.

## Referências

- [Arquitetura e organização dos pacotes](arquitetura.md)
- [Contratos de ambiente e regras](ambiente.md)
- [Contrato do agente RL](contrato-agente-rl-v1.md)
- [Contrato de observação RL v2](contrato-observacao-rl-v2.md)
- [Informações condicionadas à decisão](informacoes-de-decisao.md)
