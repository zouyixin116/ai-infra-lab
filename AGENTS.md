# Collaboration rules for AI Infra Lab

These instructions apply to every Codex session working in this repository.

## Teaching workflow

- Treat this project as a guided learning process, not a command-execution service.
- Treat each Stage as a sequence of exploration units. An exploration unit may
  be one concept, one mechanism, one guiding question, or one coherent
  exploration direction containing several closely related experiments. Its
  boundary should follow a meaningful learning narrative, not an artificially
  atomic topic and not the entire Stage.
- Explain the purpose of each experiment before asking the user to run it.
- Let the user perform meaningful commands and inspect outputs when practical.
- Explain what the output proves, what it does not prove, and how it connects to the underlying system.
- Advance one exploration unit at a time. Do not mark it complete until its
  important claims have been tested with appropriate evidence.
- When an exploration unit has enough evidence to be considered complete,
  explicitly tell the user that it is ready to summarize and remind them that they can say
  “好的，总结吧”. Do not wait until the end of the Stage to offer this reminder.

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

- Update Notion only when the user explicitly asks for an update or says an
  exploration unit is ready to summarize, such as “好的，总结吧”.
- When the user confirms that an exploration unit is ready, summarize that unit
  into the matching Stage page immediately. Do not defer completed explorations
  and combine them into one end-of-Stage summary, because that can omit
  intermediate experiments, reasoning, and problems encountered.
- Keep exploration summaries cumulative within the Stage page: preserve earlier
  exploration sections and add the newly completed exploration as its own
  section.
- Summarize the completed exploration unit into the matching Stage page; do not
  create a separate page when an existing Stage page is the intended destination.
- Include the complete exploration, not only tests suggested by Codex. Record additional experiments the user performed independently.
- Include relevant knowledge that emerges during the exploration even when it
  was not part of the original experiment plan. This includes the user's
  follow-up questions, independently raised observations, conceptual side
  questions, and any detail the user explicitly asks to “记下来”. Integrate
  each item into the section where it supports the learning narrative instead
  of dropping it merely because it was not a benchmark step.
- Capture:
  - what was explored and why;
  - commands or code that materially contributed to the exploration;
  - observed output;
  - how to interpret that output;
  - problems encountered, evidence collected, and how they were resolved;
  - conclusions and important limitations on those conclusions.
- Update the page's “当前进度” whenever an exploration unit is completed.
- Update the page's “Table of contents” whenever a new exploration section is added.
- Keep the Table of contents concise: list exploration titles only as bullets. Do not include every subsection.
- Do not add sections named “当前回答的问题” or “下一步”.
- Do not add a proposed next step to a completed-exploration summary unless the user explicitly asks for one.
- Preserve raw numeric outputs and distinguish measured facts from inference.
- Before editing an existing Notion page, fetch its current content and make a scoped update that preserves unrelated notes.

## Repository safety

- Preserve user-created or concurrently created work that is unrelated to the current request.
- Do not commit or push unrelated changes together with the current task.
- Never commit credentials, authentication files, private keys, or access tokens.
