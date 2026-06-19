# Contribuindo

Obrigado pelo interesse em contribuir! 🚀

## Como contribuir

### Reportar bugs
Abra uma issue com:
- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs observado

### Sugerir melhorias
Abra uma issue com o label `enhancement`:
- O que você quer adicionar/mudar
- Por que isso é útil
- Exemplo de uso (se aplicável)

### Pull Requests

1. Fork o repositório
2. Crie uma branch: `git checkout -b feat/nome-da-feature`
3. Faça suas alterações
4. Rode os testes: `pytest tests/ -v`
5. Commit com mensagem clara
6. Push: `git push origin feat/nome-da-feature`
7. Abra um Pull Request

## Padrões de código

- Seguimos `ruff` para linting (sem configuração extra)
- Type hints são obrigatórios em código novo
- Docstrings nos módulos e classes públicas
- Testes para código novo (pytest)

## Estrutura

```
src/
  naive/      → RAG linear (sem transformação)
  advanced/   → Query rewrite + hybrid + rerank
  agentic/    → LangGraph multi-step
```

Cada estágio é independente. Se for adicionar um novo, crie `src/novo-estagio/`.

---

Dúvidas? Abre uma issue ou manda email para rodolfo@dtsqd.com.
