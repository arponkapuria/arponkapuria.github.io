---
title: Designing an AI Coding Agent Platform like Claude Code / Cursor
description: A system design walkthrough for building the agentic coding panel behind tools like Claude Code or Cursor. This article covers execution architecture, scaling to millions of sessions, and handling failures like infinite loops, crashes, and irreversible actions.

date: August 11, 2026
modified: August 11, 2026

category: Dev Journal
tags: System Design, AI, LLM, Memory
---


## Problem Statement

The scope here is deliberately narrow: not the whole code editor, just the agent mode panel — the part where a user describes a task in plain language and the agent takes it from there, writing code, running it, reading the results, and iterating until the task is actually done.

## Requirements

### Functional Requirements

- **Task input** — the user describes a task in natural language, from either an editor extension or a CLI
- **Codebase understanding** — the agent can read files, search the codebase, and assemble the context it needs
- **Code generation and editing** — the agent writes new code and modifies existing files
- **Code execution** — the agent can run commands, build the project, run tests, start the app, and install dependencies
- **Iteration loop** — when a test fails or a build breaks, the agent reads the error and fixes it, rather than stopping
- **Checkpoints** — the user can roll back to any earlier point in the session
- **Permission control** — certain actions (deleting files, running shell commands) require explicit user approval before they happen

### Non-Functional Requirements

- **Security** — across both local and cloud/background execution, the agent must never be able to take an irreversible action on its own
- **Latency** — the first token of a response should start streaming within 1-2 seconds
- **Reliability** — a mid-task crash must never leave the codebase half-edited; changes need to be atomic and recoverable
- **Cost** — token efficiency is treated as a design constraint, not an afterthought
- **Scalability** — the design should hold up at the scale of a product like Cursor or Claude Code
- **Context limits** — the system has to select the right slice of the codebase to show the model, not the whole thing
- **Provider resilience** — if one LLM provider goes down or gets rate-limited, the system should fail over to another automatically

## High Level Design

The agent can run in two modes, depending on where the code actually lives and executes: **cloud execution** (the codebase and sandbox live on our servers — think a background agent working on a repo you've connected) and **local execution** (the codebase lives on your machine, and the agent only reasons in the cloud but touches files/terminal locally — this is how an IDE extension like Cursor or Claude Code normally works). Both share the same core idea — an orchestrator that plans, delegates, and iterates — but where the actual write/execute step happens is different, and that difference changes the security model.

### Cloud Execution Architecture

In this mode, both the "brain" (the LLM reasoning) and the "hands" (the code execution) run in our cloud — nothing touches the user's own machine.

![Cloud Execution Architecture|700](/images/blogs/system-design/cloud-execution-architecture.png)


Here's the flow, step by step:

1. **Agent Panel** — the user types a task ("add pagination to this endpoint") into the panel. This is the same UI whether it's a browser extension or a CLI.
2. **API Gateway** — every request from the client passes through here first. It doesn't do any "thinking" — it's the front door that routes requests and enforces basic request-level rules (size limits, throttling).
3. **Auth Service** — before anything runs, the gateway checks with Auth to confirm the user is who they say they are and is allowed to run agent tasks. This sits outside the main task loop because it's a one-time check per request, not something the agent needs to re-check while it's working.
4. **Orchestrator Service** — this is the core of the system, and it does three things in a loop:
   - **Plan** — break the task into steps ("find the route file, add a query param, write a test")
   - **Delegate** — hand off the actual work (writing a file, running a command) to the Sandbox
   - **Iterate** — if something fails (a test breaks, a command errors out), the orchestrator doesn't give up — it reads the failure, re-plans, and tries again. This loop is exactly what makes it an *agent* instead of a one-shot code generator.
5. **AI Gateway** — whenever the orchestrator needs the LLM to think (plan the next step, write code, interpret an error), it doesn't call OpenAI or Anthropic directly. It goes through an AI Gateway, which picks a provider and — critically — **fails over to a different provider automatically** if the first one is down or rate-limited. This directly satisfies the "provider resilience" requirement: the orchestrator doesn't need to know or care which LLM actually answered.
6. **Sandbox** — this is where code actually gets touched. It's an isolated, disposable environment where the agent can write files, run shell commands, run tests, and read files back — isolated so that nothing the agent does here can affect anything outside this one task's environment. This is also what makes the "agent must never do anything irreversible on its own" requirement enforceable: the blast radius of any mistake is contained to a throwaway sandbox, not the real system.
7. **Redis (active session history)** — while a task is actively running, the orchestrator keeps the fast-changing state (what step it's on, recent messages) in Redis, since this needs to be read and updated constantly and doesn't need to survive forever.
8. **Postgres (execution history)** — every completed step, command run, and file change is durably logged here. This is what powers the **checkpoints** requirement — rolling back to an earlier state just means restoring from a recorded point in this history, and it's also what lets the system recover cleanly if a task crashes mid-way instead of leaving the codebase half-edited.

The reason session state and execution history are two separate stores, not one: they're read/written at very different rates and need very different guarantees. Session history changes on every single step and just needs to be fast and available — losing a small amount on a crash is tolerable. Execution history is the source of truth for checkpoints and recovery, so it needs to be durable, even if it's slightly slower to write to.

### Local Execution Architecture

This mode is used when the agent needs to work on a codebase that lives on the user's own machine — the far more common case for a coding assistant, since most code isn't sitting on our servers.

![Local Execution Architecture|700](/images/blogs/system-design/local-execution-architecture.png)

The split here is between **Cloud** (where the thinking happens) and **Local Machine** (where the actual files, terminal, and codebase live):

1. **Agent Panel** — same as before, the user describes a task from their editor or CLI. This sends the task up to the cloud.
2. **Orchestrator Agent (cloud)** — plans, delegates, and iterates, exactly like the cloud version. But now it has two extra cloud-side helpers:
   - **Context engine** — since the agent can't just "see" the user's whole codebase, this builds a map of the repo and searches code to figure out which files are actually relevant to the task. This is what solves the "context limits" requirement — picking the right slice of code instead of trying to stuff the whole repo into the model.
   - **Session store** — keeps conversation history and checkpoints, the same role Redis/Postgres played in the cloud version.
   - **Sub-Agent** — for pure exploration work (like "search the codebase for where this function is used"), the orchestrator can spin off a sub-agent with its own isolated context, so exploring a large codebase doesn't clutter or blow up the main task's context window.
3. **AI Gateway** — same role as before: routes LLM calls to OpenAI, Anthropic, or Google, with automatic failover.
4. **Delegate subtask (to the local machine)** — when the orchestrator actually needs to *do* something (write a file, run a test), it can't do that itself — it sends that specific subtask down to the user's machine.
5. **Permission Scoped Layer** — this is the most important box in this diagram, and it's the direct answer to the "permission control" and "security" requirements. Every subtask sent from the cloud passes through this layer *before* it's allowed to touch anything local. It enforces what the agent is and isn't allowed to do — for example, editing a file might be auto-approved, but deleting a file or running an arbitrary shell command might require the user to explicitly approve it first. This is labeled as a **security trade-off** in the diagram because there's a real tension here: too many permission prompts and the agent feels annoying to use; too few and the agent can do something destructive on a real, non-disposable codebase. Unlike the cloud sandbox (which is disposable and isolated by default), the local machine is the user's actual environment — so permission scoping has to do the job that sandboxing does in the cloud version.
6. **write file / run commands / run tests / read files** — the actual actions, executed against the real **filesystem**, **terminal**, and **codebase** on the user's machine, once the permission layer clears them.
7. **Output/error flows back up** to the orchestrator, which — same as the cloud version — iterates if something failed, re-planning and trying again.

**The key architectural difference between the two modes:** in cloud execution, safety comes from *isolation* — the sandbox is disposable, so mistakes are cheap. In local execution, isolation isn't possible — you can't sandbox someone's actual laptop — so safety instead comes from *permission scoping*, checking every action against what the user has allowed before it touches anything real. Same orchestrator, same plan-delegate-iterate loop, but two different safety mechanisms because the two environments have fundamentally different risk profiles.

## Scaling the Architecture

The high-level design above works, but it has a hidden assumption: everything happens in one place, one session at a time. At real scale — millions of users, thousands of agents running concurrently — a few things break, and the architecture below is the fix. Same core loop (plan, delegate, iterate), but every piece is now built to survive bursts, crashes, and cost pressure.

![AI Coding Agent Platform Architecture at Scale|700](/images/blogs/system-design/ai-coding-agent-platform-architecture-at-scale.png)

### Client

**Agent Panel/CLI** — same entry point as before, just labeled here as "millions of users" to make explicit what this design has to hold up under.

### Cloud Platform

**Message Queue + Session Worker Pool**

The first new piece, and it solves a problem the earlier design didn't have to deal with: what happens when a burst of task requests arrives faster than the orchestrators can handle them? Instead of a client's request hitting an orchestrator directly, it first lands in a **Message Queue**. A pool of **Session Workers** then pulls requests off the queue at whatever pace the system can actually handle. This decouples *how fast requests come in* from *how much capacity is available to process them* — during a burst, requests simply wait a little longer in the queue instead of overwhelming the orchestrators or getting dropped.

**Orchestrators (stateless, run as durable workflow steps)**

This is the most important shift from the high-level design. Earlier, the orchestrator was one long-running process holding a task's state in memory. At scale, that's fragile — if that process crashes on step 30 of a task, everything is lost and the task restarts from zero. Here, the orchestrator is **stateless**: each step of the plan-delegate-iterate loop is saved as it happens, through a workflow engine. If the process crashes on step 30, a new orchestrator instance picks up exactly at step 30, not step 0. This is what makes the "reliability — a crash mid-task should not leave the codebase half-edited" requirement actually hold at scale, not just in theory.

Underneath this sits an important rule, called out directly in the diagram: **the orchestrator never talks to Redis, Postgres, or blob storage directly.** It only ever calls a **Workflow SDK** to save or load its state. Why this matters: if every orchestrator instance were writing directly to the database, any change to how state is stored would mean changing code in every orchestrator. Going through one SDK means the storage layer underneath can change (add a cache, change databases, shard it) without touching the orchestrator logic at all.

- **Redis (hot state)** — the fast-changing, short-lived part of a session (what step it's on right now)
- **Postgres (durable records)** — the permanent record of what happened, used for recovery and audit

**Sub-Agents (isolated context)**

When the main orchestrator needs to explore the codebase (search for a function, understand how a module works), it doesn't do that exploration in its own context. It delegates it to a **Sub-Agent** with its own separate, isolated context window, and only gets a **summary back** — not the full raw exploration transcript. This keeps the main loop's context clean and small, which directly helps with both the "context limits" and "cost/token efficiency" requirements — exploration can get noisy and use a lot of tokens, and none of that noise needs to sit in the main task's context forever.

**AI Gateway (failover + cost routing)**

Same failover role as before (if OpenAI is down, try Anthropic or Google), but now it does one more job: **cost routing**. Not every LLM call in the loop needs the most expensive model — a planning step ("what should I do next?") can often use a cheaper, faster model, while the actual code-writing step might need a stronger one. Routing each call to the cheapest model that can do the job is a direct, practical answer to the "token efficiency is a design constraint" requirement.

**Prompt Cache**

Sitting next to the AI Gateway, this caches repeated or overlapping prompt content (like a shared system prompt or a repeated tool description) so the same tokens don't need to be sent and processed by the LLM provider over and over. This is a straightforward win for both cost and latency, especially since many calls in one session — and across sessions — share large, identical chunks of prompt.

**Storage Service + Blob Storage**

The orchestrator's full conversation logs and filesystem snapshots are persisted here, separately from the fast-moving Redis/Postgres state. This is what checkpoints are actually built from — a snapshot in Blob Storage is a point the user can roll back to.

### Guardrails Layer

Every tool call the orchestrator wants to make — whether it's a harmless read (`read_file`, `search`) or something riskier (edit a file, run a command, run tests) — passes through a **Guardrails layer** before it's allowed to reach actual execution. This is the single choke point where the system decides: is this action auto-reversible (just do it), auto-irreversible-but-allowed (log it carefully), or does it need a human to explicitly approve it first? Centralizing this decision in one layer — rather than scattering permission checks across every tool — means the rules can be audited and changed in one place.

### Execution

This is where the actual work happens, and it looks different depending on where the code lives.

**Local Machine**

When the codebase is on the user's own computer, the guardrails layer's decision is enforced by a **Permission Scope Layer** (an allowlist/denylist of what the agent can do), before any tool touches the real filesystem or terminal. The diagram is direct about the trade-off here: this is a **tiny security boundary, with no real sandbox** — the user's own machine *is* the sandbox, for better or worse. That's exactly why the permission layer has to do more work here than it does in the cloud case: there's no disposable environment to contain a mistake, so the only real protection is stopping the risky action before it happens.

**Sandbox Pool (pre-warmed)**

When execution happens in the cloud instead, the system uses a pool of **pre-warmed microVMs** — one sandbox per active session. "Pre-warmed" matters for latency: spinning up a fresh isolated VM from scratch for every single task would blow past the "first token in 1-2 seconds" requirement; keeping a pool ready to assign means a session can start executing almost immediately. Each session gets its own microVM, so one task's mistake — a bad `rm`, an infinite loop — can't touch another user's session.

**Checkpoint Layer (shadow git)**

Inside the sandbox, every change is also tracked through a shadow git layer, which is what actually implements the "roll back to any earlier state" requirement at the filesystem level — think of it as an automatic, invisible version history running alongside the agent's edits, independent of whatever git repo the user's own code might already have.

### Why This Design Holds Up at Scale

Three ideas repeat throughout this diagram, and they're worth stating explicitly as the "why" behind the whole thing:

1. **Decouple intake from capacity** — the message queue absorbs bursts so the orchestrators are never hit with more than they can handle at once
2. **Make state recoverable, not just held in memory** — stateless orchestrators plus a workflow engine mean a crash costs a few seconds of replay, not a lost task
3. **Put safety at one choke point, not scattered everywhere** — the guardrails layer and the permission/sandbox split are both single, centralized places where risky actions are decided on, rather than trusting every individual tool to enforce its own rules

## Handling Failures

- **Infinite reasoning loop** — the agent gets stuck bouncing between planning and execution, never converging, silently burning tokens. Fix: a hard cap on iterations, with detection for repeated identical tool calls — once caught, stop and tell the user to try a different approach instead of continuing to burn tokens.

- **Hung execution** — a command the agent runs never returns (a server that never exits, a process waiting on input that never comes), and the orchestrator sits blocked waiting on it. Fix: a timeout on every command and background process, with the session killed and restarted cleanly once that timeout hits.

- **Mid-session crash** — the process dies mid-edit, leaving files half-changed, violating the requirement that a task either resumes cleanly or fully reverts. Fix: atomic file operations, with a checkpoint taken before every edit and validated after. If a crash happens mid-operation, the system rolls back to the last valid checkpoint rather than leaving a partial edit in place.

- **Irreversible actions slipping through** — even with guardrails, the agent could still attempt something destructive (dropping a table, deleting files outside the workspace, force-pushing to a remote). Fix: an explicit action-classification step inside the guardrails layer, backed by a maintained list of irreversible operations — anything on that list is never auto-executed and always requires mandatory human approval first.

---

> *This article is a written walkthrough of this video - [Design Claude Code Like a Senior Engineer](https://www.youtube.com/watch?v=YtRIj_KU9d4)*