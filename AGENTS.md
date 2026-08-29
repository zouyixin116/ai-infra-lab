# Collaboration rules for AI Infra Lab

These instructions apply to every Codex session working in this repository.

## Teaching workflow

- Treat this project as a guided learning process, not a command-execution service.
- Explain the purpose of each experiment before asking the user to run it.
- Let the user perform meaningful commands and inspect outputs when practical.
- Explain what the output proves, what it does not prove, and how it connects to the underlying system.
- Advance one concept at a time. Do not mark a concept complete until its important claim has been tested with appropriate evidence.

## Code comments

- Write code comments in English.
- Add comments where they reveal intent or behavior that is not obvious from the syntax.
- Use short inline comments at critical trigger points, including:
  - a backward call that triggers DDP hooks or NCCL gradient collectives;
  - DDP construction that broadcasts rank 0 parameters;
  - an optimizer step that applies already-synchronized gradients locally;
  - CUDA synchronization used to make timing valid;
  - rank-local data partitioning and deterministic sampler behavior;
  - rank-specific checkpoint writing, barriers, and cross-rank verification.
- Use paragraph comments immediately before a block when several lines implement one design decision.
- Do not add comments that merely restate the syntax.
- Keep comments technically precise. For example, do not imply that `optimizer.step()` performs gradient AllReduce when synchronization actually occurs during DDP backward.

## Notion learning notes

- Update Notion only when the user explicitly asks for an update or says a concept is ready to summarize, such as “好的，总结吧”.
- Summarize the completed concept into the matching Stage page; do not create a separate page when an existing Stage page is the intended destination.
- Include the complete exploration, not only tests suggested by Codex. Record additional experiments the user performed independently.
- Capture:
  - what was explored and why;
  - commands or code that materially contributed to the exploration;
  - observed output;
  - how to interpret that output;
  - problems encountered, evidence collected, and how they were resolved;
  - conclusions and important limitations on those conclusions.
- Update the page's “当前进度” whenever a concept is completed.
- Update the page's “Table of contents” whenever a new exploration section is added.
- Keep the Table of contents concise: list exploration titles only as bullets. Do not include every subsection.
- Do not add sections named “当前回答的问题” or “下一步”.
- Do not add a proposed next step to a completed-concept summary unless the user explicitly asks for one.
- Preserve raw numeric outputs and distinguish measured facts from inference.
- Before editing an existing Notion page, fetch its current content and make a scoped update that preserves unrelated notes.

## Repository safety

- Preserve user-created or concurrently created work that is unrelated to the current request.
- Do not commit or push unrelated changes together with the current task.
- Never commit credentials, authentication files, private keys, or access tokens.
