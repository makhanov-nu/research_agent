# Discord commands

The bot replies in DMs and when **@-mentioned** in a channel. Anything not
starting with `!` is treated as a message to the agent. Commands:

| Command | Description |
| --- | --- |
| `!help` | Show the command list |
| `!remember <text>` | Store a durable preference/instruction (procedural memory) |
| `!checkpoint` (`!summarize`) | Summarize this thread to long-term memory and reset the live context |
| `!ideate <topic>` | Convene the [scored consortium](consortium.md): one round, every surviving idea ranked with author + bibliography |
| `!tasks` | List recent subagent tasks (the dashboard) |
| `!task <id>` | A task's status + result |
| `!trace <id>` | Export a task's full trace (reasoning + tool calls) as a file |
| `!getfile <path>` | Fetch a written output (e.g. a drafted LaTeX review) into Discord |
| `!runs` | List experiments and their status |
| `!approve <id>` | Approve and launch a pending experiment |
| `!cancel <id>` | Cancel a running experiment |

!!! note
    Experiment commands require a configured compute node; consortium requires
    `OPENROUTER_API_KEY`; task/memory commands require the database. When a
    capability isn't configured, the command says so.

## Artifacts

Written outputs live under `outputs/` on the server:

- `outputs/lit_reviews/*.tex` / `*.bib` — drafted literature reviews
- `outputs/ideas/*.md` — consortium transcripts
- `outputs/traces/task_*.json` — exported task traces

Retrieve any of them with `!getfile <relative-path>` (path-confined to `outputs/`).
