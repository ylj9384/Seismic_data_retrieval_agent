# 多智能体地震数据处理系统说明

本项目包含多智能体协作的地震数据检索与分析系统，支持人机交互（Human-in-the-Loop）、多任务编排、智能工具调用，并包含自演化测试模块。系统主要由以下四个部分组成：

---

## 1. data_retrieval

**功能：**  
- 地震数据检索智能体，负责台站、波形、事件等地震相关数据的查询、下载、绘图等操作。
- 支持参数澄清、主动追问、批量数据处理等 Human-in-the-Loop 能力。
- 主要文件：
  - `agent_initializer.py`：定义数据检索智能体的节点流程和工具注册。
  - `nodes.py`：LLM节点与工具节点实现，负责参数解析、工具调用、澄清追问等。
  - `prompt_templates.py`：LLM提示词模板，强化多轮交互和澄清能力。
  - `tools.py`、`tool_registry.py`：具体工具实现与注册。
  - `state.py`：智能体状态管理。

---

## 2. phase_detection

**功能：**  
- 地震相位检测智能体，负责地震波形的自动识别、相位标注、相关工具调用等。
- 支持多轮交互、参数补全、工具链式调用。
- 主要文件：
  - `agent_initializer.py`：定义相位检测智能体的节点流程和工具注册。
  - `nodes.py`：LLM节点与工具节点实现，负责参数解析、工具调用、澄清追问等。
  - `prompt_templates.py`：LLM提示词模板。
  - `tools.py`、`tool_registry.py`：具体工具实现与注册。
  - `state.py`：智能体状态管理。

---

## 3. orchestrator

**功能：**  
- 主编排器（Orchestrator），负责多智能体的路由、任务分配、上下文维护和最终输出。
- 实现 Human-in-the-Loop 机制，支持参数澄清、计划确认、结果反馈等多轮交互。
- 主要文件：
  - `agent_supervisor.py`：主流程编排，路由决策、Agent执行、HITL交互、上下文维护。
  - `dialogue_manager.py`：对话历史与上下文管理。
  - `clarification.py`：澄清追问生成。
  - `state.py`、`schemas.py`：全局状态与数据结构定义。
  - `system.py`：主入口与系统集成。

---

## 4. z_self_evolving_test

**功能：**  
- 智能体自演化与工具动态加载测试模块。
- 支持自定义工具、公式检索、RAG（检索增强生成）、公式库管理等。
- 主要文件：
  - `llm_chat.py`：自演化智能体主逻辑，支持多轮对话与工具调用。
  - `prompt_templates.py`：自演化智能体的提示词模板。
  - `dynamic_tools/`：动态工具目录，包含多种地震相关计算与查询工具。
  - `formula_store/`：公式库与检索相关文件。
  - `tool_runtime/`：工具运行环境与解析器。

---

## 使用说明

1. **主流程入口**  
   - 通过 orchestrator 的 `system.py` 或 `main.py` 启动主编排器，输入地震相关问题，系统会自动分配合适智能体并多轮交互完成任务。

2. **参数澄清与多轮交互**  
   - 当参数不全或需求不明时，系统会主动追问用户，等待补充后继续任务，实现人机协同。

3. **自演化测试**  
   - 进入 `z_self_evolving_test` 目录，运行 `llm_chat.py` 可体验智能体自演化与工具动态加载能力。

---

## 目录结构简述

- `data_retrieval/`：地震数据检索智能体
- `phase_detection/`：地震相位检测智能体
- `orchestrator/`：主编排器与多智能体协作
- `z_self_evolving_test/`：自演化与工具动态加载测试

---

## 适用场景

- 地震数据检索与分析
- 多智能体协作任务
- 人机交互式数据处理
- 智能工具链式调用与自演化测试

---

如需详细用法或扩展说明，请查阅各目录下的 `readme` 或源码注释。
