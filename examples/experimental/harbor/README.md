# Harbor in-process on cloud sandboxes

This example runs [Harbor](https://github.com/harbor-framework/harbor) trials
**inside the rollout worker**: the agent function builds a `TrialConfig` and
calls `Trial.run()` directly, with the task sandbox on a cloud
[sandbox provider](../../../docs/user-guide/sandbox-providers.md) the worker
reaches over the network. There is no agent server.

Compared with [`examples/swe-agent-harbor-docker`](../../swe-agent-harbor-docker/README.md):

| | agent server (`swe-agent-harbor-docker`) | in-process (this example) |
| --- | --- | --- |
| Where `Trial.run()` runs | a separate host with a Docker daemon | the rollout worker |
| Sandbox backends | `docker` (local), `daytona` via the server's env | any cloud sandbox provider the worker can reach |
| Moving parts | trainer → HTTP → agent server → Harbor | trainer → Harbor |
| Use it when | tasks must run on the local Docker daemon | sandboxes are cloud-hosted |

Everything on the trainer side is the same: TITO, the session server, GRPO, the
reward hook (`generate.py` from the agent-server example).

## 1. Install Harbor in the rollout environment

Harbor now runs where the rollout workers run, so it goes into the Miles
image / environment. Use the `harbor-miles-v0.20.0` branch of
`harbor-framework/harbor` — the terminus-2 truncation policy it carries is
required for TITO (see the agent function's header for the full list) —
with the extra for your backend:

```bash
# uv, not pip: the branch carries a uv-workspace dependency pip cannot resolve
uv pip install "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
# or harbor[daytona], harbor[modal], ...
```

## 2. Provision the sandbox backend

Credentials and endpoints for every provider are in
[Sandbox Providers](../../../docs/user-guide/sandbox-providers.md). The launch
command below uses E2B:

```bash
mkdir -p ~/.config/e2b && echo e2b_... > ~/.config/e2b/api_key
```

Task directories: `HARBOR_TASKS_DIR` must contain one Harbor task dir per
`metadata.instance_id` in the training data (same as the agent-server example);
put it on a filesystem every worker can read.

**Network.** In-sandbox agents (mini-swe-agent, claude-code) call the model
from inside the sandbox, so the sandbox platform must reach the Miles session
server. `--session-server-external-host` is the address the sandboxes reach it
on: the driver builds the URL the agent gets from it and keeps the session
servers on the head node. On a multi-node job see the network note in
[the agent-server example](../../swe-agent-harbor-docker/README.md). Two port
ranges must route from the sandbox network: one
session-server port per worker starting at `--session-server-port` (30000-30031
for `run.py`'s 32 workers) and the SGLang router's 31000. Host-process agents
(terminus-2) call the model from the worker and need no sandbox egress.

## 3. Prepare data

Same as the agent-server example:

```bash
python examples/swe-agent-harbor-docker/download_and_process_data.py \
    --input /path/to/terminal-bench.jsonl \
    --output /path/to/tb2_train.jsonl \
    --agent-name mini-swe-agent \
    --prompt-key instruction
```

## 4. Launch

```bash
HARBOR_ENV_TYPE=e2b python examples/experimental/harbor/run.py \
    --num-nodes 1 --num-gpus-per-node 8 --skip-prepare \
    --megatron-path /root/Megatron-LM \
    --hf-checkpoint /path/to/GLM-4.7-Flash \
    --ref-load /path/to/GLM-4.7-Flash_torch_dist \
    --save-dir /path/to/checkpoints \
    --prompt-data /path/to/tb2_train.jsonl \
    --harbor-tasks-dir /path/to/harbor_tasks \
    --session-server-external-host <trainer-address-reachable-from-the-sandboxes> \
    --rollout-batch-size 4 --n-samples-per-prompt 8 --global-batch-size 32 \
    --num-rollout 200 --save-interval 10
```

`--save-dir` needs real headroom: a GLM-4.7-Flash torch_dist checkpoint with
optimizer state is several hundred GB, and the end-of-run save will fill
whatever is there.

`HARBOR_ENV_TYPE` has no default: the backend decides whose quota a run spends.
Backend-specific settings go in `HARBOR_ENV_KWARGS` as a JSON object (Harbor's
`EnvironmentConfig.kwargs`), e.g. `'{"auto_snapshot": true}'` for Daytona.

`HARBOR_OVERRIDE_CPUS`, `HARBOR_OVERRIDE_MEMORY_MB`, and
`HARBOR_OVERRIDE_STORAGE_MB` replace a task's declared resource values. They do
not change how Harbor enforces CPU or memory. Set
`HARBOR_CPU_ENFORCEMENT_POLICY` or `HARBOR_MEMORY_ENFORCEMENT_POLICY` to one of
Harbor's `auto`, `request`, `limit`, `guarantee`, or `ignore` modes when the
provider default is not the desired contract. For example, Modal CPU `auto` is
a hard request-and-limit pair; use `HARBOR_CPU_ENFORCEMENT_POLICY=request` to
keep the requested CPU guarantee while allowing burst capacity.

Every remaining knob — timeouts and their layering, failure semantics, the
full env-var reference — is documented in `harbor_agent_function.py`'s header,
next to the code that reads it.

## Validation

This README's command has been run end to end: 8×H200, real training mode with
the batch dials reduced, both trials scoring reward 1.0 and one GRPO step
completed. Which sandbox providers this path has been run on is the table in
[Sandbox Providers](../../../docs/user-guide/sandbox-providers.md).

## Qwen3.8-27B on Terminal-Bench 2.1

[`run_qwen38_27b.py`](./run_qwen38_27b.py)
is a full-parameter, fully asynchronous GRPO example for **8 nodes of 8 H200s**:
2 trainer nodes and 6 rollout nodes. It uses this example's in-process Harbor
integration, E2B sandboxes, and Terminus 2/XML. There is no separate agent server.
The script creates neither machines nor Kubernetes jobs: provision an allocation
and use the [Ray launcher workflow](../../../docs/getting-started/quick-start.md).

This is an example adapted from a previous agentic recipe, **not a new
end-to-end validated TB2.1 training result**. Training on benchmark tasks
contaminates that benchmark: do not report subsequent TB2.1 performance as
held-out generalization. Keep a separate evaluation set.

### Prepare the environment and task data

Use a Miles image with compatible SGLang, Megatron-LM, and Qwen3.8 support.
Install Harbor as described above on every rollout host, then install the
small recipe-specific dependency:

```bash
uv pip install --no-deps -e .
uv pip install -r examples/experimental/harbor/requirements-qwen38.txt
git clone https://github.com/harbor-framework/terminal-bench-2-1.git /scratch/tb21
python examples/experimental/harbor/prepare_terminal_bench.py \
    --tasks_dir /scratch/tb21/tasks --output /scratch/tb21/train.jsonl
```

This uses the [official TB2.1 task repository](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1),
not TB2.0. Pin and record the task checkout's commit for reproducible runs.
The task directory must directly contain each task's `task.toml` and
`instruction.md`. The helper writes one task-ID prompt per task, adds
`agent_name=terminus-2`, records instruction/config hashes, and refuses to
overwrite an existing JSONL. It does not expose solutions or test answers.
Use `--exclude_task TASK_A TASK_B` for an explicit held-out split or tasks your
E2B provider cannot support (for example GPU or multi-container requirements).
Check provider capabilities before training; do not silently score unsupported
environments as model failures.

Stage the checkpoint at `/scratch/models/Qwen3.8-27B` and the same task/data
paths on every node, using shared storage or explicit synchronization. Mount an
E2B key file at the same private path on all rollout hosts. Custom E2B endpoints
can be configured with `E2B_API_URL` and `E2B_SANDBOX_URL`; do not embed keys in
the script, dataset, or repository. Allow cross-node communication to the
session-server ports and SGLang router. On an existing Ray allocation set
`MILES_SCRIPT_EXTERNAL_RAY=1` before launch. This recipe **requires** that mode
so the shared launcher's fresh-host Ray/SGLang cleanup cannot interrupt an
existing cluster. Join all allocated nodes to the same Ray cluster first.

### Inspect, then launch

Run from the Miles checkout. This CLI uses Tap's underscore-style flag names:

```bash
python examples/experimental/harbor/run_qwen38_27b.py \
    --dry_run --model_dir /scratch/models --data_dir /scratch/tb21 \
    --output_dir /scratch/runs

# Run only after inspecting the proposed configuration and provisioning nodes.
export MILES_SCRIPT_EXTERNAL_RAY=1
python examples/experimental/harbor/run_qwen38_27b.py \
    --model_dir /scratch/models --data_dir /scratch/tb21 \
    --output_dir /scratch/runs --megatron_path /opt/Megatron-LM \
    --e2b_api_key_file /secrets/e2b.key
```

The second command first converts the supplied HF checkpoint using Miles'
Qwen3.8 model definition. Pass `--skip_prepare` if the matching conversion
already exists at `/scratch/models/Qwen3.8-27B_torch_dist`. A new run ID starts
from those base weights; to resume, supply the original `--run_id` and output
directory. Never reuse a run ID for a different experiment.

The authoritative defaults live in `ScriptArgs` in the launcher. The baseline
uses 8 prompts × 32 independent episodes = 256 trajectories per update,
131,072-token sequence and 49,152-token per-turn response limits, TP4/CP2 on the
trainers, dynamic token batching, full activation recomputation, CPU optimizer
offload, and a constant learning rate of 1e-6. It allows 128 concurrent
episodes, uses cache-aware routing, and corrects async policy lag with clipped
importance sampling. Trainer logprobs are recomputed; it does **not** enable
`--use-rollout-logprobs`. Host-memory HiCache is not enabled by this baseline.

Summarization is explicitly **off**, matching the append-only training recipe.
Do not turn it on by changing an environment variable: the launcher pins it
off. A summarizing recipe needs session server v2 plus dynamic global batching
and separate validation of segment/reward accounting. The per-trial agent
timeout is eight hours, with another 30 minutes for setup/verifier cleanup;
native verifier limits/resources remain in the task configs.

Checkpoints are saved every 25 updates over 1,000 updates. All outputs go under
`<output_dir>/<run_id>/`: `checkpoints/`, Miles `traces/`, and complete
`harbor_trials/`. Keep those paths on a large disk (and collect node-local
artifacts from every rollout host if they are not shared). A checkpoint with
optimizer state is hundreds of GiB; arrange retention and external backups.
The dashboard is enabled; W&B is enabled by the shared launcher when
`WANDB_API_KEY` is set. No API keys should be passed as command-line arguments.
