# 切换 LLM 模型

所有使用 LLM 的 Agent（Analysis、Tuning、Report、MCTS 策略等）都通过 `utils.llm_factory.get_llm()` 创建模型，内部使用 LangChain 的 `init_chat_model`，只需修改 config 即可切换。

## 快速切换

在 `main.py` 或 config 中修改：

```python
# OpenAI（默认）
config["llm_provider"] = "openai"
config["llm_model"] = "gpt-4o"

# Google Gemini
config["llm_provider"] = "google"
config["llm_model"] = "gemini-2.0-flash"

# Anthropic Claude
config["llm_provider"] = "anthropic"
config["llm_model"] = "claude-3-5-sonnet-20241022"
```

## 环境变量

| Provider | 环境变量 | 安装包 |
|----------|----------|--------|
| OpenAI | `OPENAI_API_KEY` | langchain-openai（已装） |
| Google | `GOOGLE_API_KEY` | `pip install langchain-google-genai` |
| Anthropic | `ANTHROPIC_API_KEY` | `pip install langchain-anthropic` |

## 可选配置

```python
config["llm_temperature"] = 0.2    # 温度 0-1
config["llm_max_tokens"] = 4000    # 最大输出 token
config["llm_api_base"] = "https://..."  # 自定义 API 端点（如代理）
config["llm_api_key"] = "sk-..."   # 覆盖环境变量中的 key
```

## 预设模型（config/default_config.py）

`LLM_CONFIG` 中定义了各 provider 的默认模型和参数，可在此扩展或修改预设。
