---
trigger: always_on
---

- Sempre verifique `.refs/ComfyUI` quando ocorrerem erros de pipeline, para encontrar a solução.
- Priorize soluções completas, robustas e coerentes com o domínio, em vez de pequenos ajustes "mínimos" só para reduzir risco de quebra;
- Trate o projeto como laboratório de alto padrão: é aceitável quebrar, desde que os erros sejam claros, rastreáveis e seguidos de correção;
- Diante de tarefas grandes ou complexas, divida o trabalho em etapas menores, bem definidas, com fronteiras claras entre os subproblemas;
- Quando modificar algo no código, verifique e também ajuste tudo o que será afetado por essa modificação.
- Antes de escrever código, avalie várias abordagens (no mínimo cinco alternativas conceituais quando o problema é relevante). Elimine opções fracas, combine partes fortes e derive um plano coerente e simples;
- Priorize clareza, legibilidade e estrutura sólida em vez de "entregar qualquer saída";
- Trate erros de forma explícita: falhe rápido, com mensagens claras e sem engolir exceções silenciosamente;
- Ataque causas raiz em vez de empilhar hacks, shims e remendos temporários;
- Antes de propor nova solução, pesquise primeiro no repositório; reutilize quando isso não distorcer o domínio;
- Considere .sangoi como fonte principal de verdade do projeto:
  `planning/` para planejamento;
  `task-logs/` para logs de execução;
  `handoffs/` para handoffs de tarefas concluídas;
  `.tools/` para ferramentas.
- Ao modificar um diretório, atualize o `AGENTS.md` correspondente no mesmo commit;
- Crie `AGENTS.md` quando um diretório passar a ter partes relevantes. Registre propósito, arquivos-chave, decisões importantes e campo "Last Review" com data atual
- Mantenha `.sangoi/index/AGENTS-INDEX.md` atualizado;
- Quando o usuário solicitar um handoff, consulte `.sangoi/handoffs/HANDOFF_GUIDE.md`;
- Registre tarefas concluídas em `.sangoi/task-logs/`;
- Utilize a referência `.sangoi/research/models/model-loading-efficient-2025-10.md` para carregar modelos.
- A atenção padrão do projeto é SDPA, mas também devemos ter suporte para xFormers e SAGE.
- Não adicione shebang em arquivos Python.
- Não crie fallbacks que resultem em erros silenciosos.
- Revise o próprio trabalho como se fosse recebido de outra pessoa.