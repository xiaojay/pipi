# Pipi

这是一个可以单独拿出去使用的 Python 项目。

它最初来自这个 monorepo 里对 `packages/coding-agent` 核心链路的重写，现在已经整理成独立代码库结构：

- `pipi/`：实际 Python 包
- `scripts/`：外围辅助脚本
- `tests/`：项目内测试
- `pyproject.toml`：项目元数据

当前它仍然是“核心能力优先”的版本，不是原版 `pi` 的完整等价移植。

当前目标不是完整复刻原版 `pi`，而是先把最核心、最可运行的一条链路落下来：

- 基于 OpenAI-compatible 接口和 Claude Messages API 的 agent 循环
- 工具调用
- JSONL 会话持久化
- 简单 CLI / REPL

## 当前已经实现

- OpenAI-compatible LLM 客户端
- Anthropic Claude Messages API 客户端
- 支持工具调用的 agent 主循环
- JSONL session 持久化
- 基于 `id` / `parentId` 的树形会话结构
- 内置工具：
  - `read`
  - `write`
  - `edit`
  - `bash`
  - `grep`
  - `find`
  - `ls`
- `@file` 形式的文件附件
- 简单交互命令：
  - `/help`
  - `/quit`
  - `/new`
  - `/session`
  - `/tools`
  - `/model <id>`
- shell 快捷方式：
  - `!command`：执行命令并把结果写入上下文
  - `!!command`：执行命令但不写入上下文

## 还没实现

下面这些能力在 TypeScript 版本里存在，但这个 Python 版本目前还没有移植：

- TUI
- extension runtime
- skills / prompt templates / themes
- OAuth 登录流
- RPC mode
- 自动 compaction 和 summarization
- 完整的多 provider / model registry
- 与 `@mariozechner/pi-ai` 的全量能力对齐

## 运行前提

这个版本当前支持两类接口：

- OpenAI-compatible Chat Completions
- Anthropic Claude Messages API

你需要满足下面任一条件：

1. 有可用的 OpenAI-compatible 在线接口
2. 有本地兼容服务，例如 Ollama、vLLM、LiteLLM 等
3. 有可用的 Anthropic API key

如果接口不支持 function calling / tool calling，这个 agent 只能退化为普通对话，无法完整使用工具链。

## 运行方式

先进入这个项目目录：

```bash
cd pipi
```

然后执行：

```bash
# 查看帮助
python3 -m pipi --help

# 单次执行
python3 -m pipi --print "Summarize this repository"

# 使用 Claude API
python3 -m pipi \
  --provider anthropic \
  --model claude-sonnet-4-5 \
  --print "Summarize this repository"

# 指定本地兼容接口
python3 -m pipi \
  --provider local \
  --model llama3.1 \
  --base-url http://localhost:11434/v1

# 进入交互模式
python3 -m pipi
```

如果使用 OpenAI：

```bash
export OPENAI_API_KEY=...
python3 -m pipi --print "Explain this repository"
```

如果使用 Claude API：

```bash
export ANTHROPIC_API_KEY=...
python3 -m pipi \
  --provider anthropic \
  --model claude-sonnet-4-5 \
  --print "Explain this repository"
```

如果你想作为独立项目安装：

```bash
cd pipi
pip install -e .
```

安装后也可以直接使用：

```bash
pipi --help
```

## 目录说明

- `pipi/agent.py`：agent 主循环
- `pipi/llm.py`：OpenAI-compatible / Claude API 客户端
- `pipi/session.py`：JSONL 会话和树结构管理
- `pipi/cli.py`：命令行入口和 REPL
- `pipi/tools/`：内置工具实现
- `pipi/file_args.py`：`@file` 文件附件处理
- `scripts/view_sessions.py`：session 浏览脚本
- `tests/`：项目测试

## 当前定位

更准确地说，这不是“原版 pi 的 Python 完整替代品”，而是：

- 一个已经能跑起来的 Python 核心版
- 一个后续继续补齐 TUI、扩展系统和 provider 适配的基础实现
- 一个便于研究 `packages/coding-agent` 主链路的最小实现
