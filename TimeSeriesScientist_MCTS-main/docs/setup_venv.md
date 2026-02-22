# 虚拟环境配置（含移动项目后重建）

## 移动项目后原来的 venv 能用吗？

**不建议继续用。** 虚拟环境在创建时会把**绝对路径**写进 `pyvenv.cfg` 和 `bin/` 里各脚本的 shebang。项目路径一变，这些路径就失效，可能出现：

- `python` / `pip` 找不到或指向错误位置  
- 激活脚本 `activate` 里路径错误  

所以**移动或重命名项目目录后，建议删除旧 venv 并重新创建**。

## 在当前项目目录下重新配置虚拟环境

在**项目根目录**（即 `TimeSeriesScientist`）下执行：

### 1. 删除旧虚拟环境（可选但推荐）

```bash
cd /Users/applebuy/agent\ sbu/TimeSeriesScientist
rm -rf .venv
```

### 2. 用系统 Python 创建新 venv

```bash
# 确保用的是 Python 3.8+
python3 -m venv .venv
```

### 3. 激活虚拟环境

```bash
source .venv/bin/activate   # macOS / Linux
```

激活后终端提示符前会出现 `(.venv)`。

### 4. 安装依赖

```bash
# 在项目根目录下安装 time_series_agent 的依赖
pip install -r time_series_agent/requirements.txt
```

### 5. 运行前设置 PYTHONPATH（若在项目根运行）

```bash
# 在项目根运行 main 或 quick test 时，把 time_series_agent 包加入路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m time_series_agent.main
# 或
python -m time_series_agent.run_quick_test
```

也可以先 `cd time_series_agent` 再 `python main.py`（视你项目入口而定）。

### 6. 配置 API Key

在项目根目录的 `.env` 中设置（或 `export`）：

- `GOOGLE_API_KEY`（若用 Gemini）
- `OPENAI_API_KEY`（若用 GPT）

---

## 使用 Conda 的写法（与 README 一致）

若你更习惯用 Conda：

```bash
conda create -n TSci python=3.11 -y
conda activate TSci
pip install -r time_series_agent/requirements.txt
```

运行方式同上，注意在项目根或 `time_series_agent` 下并设置好 `PYTHONPATH` / `.env`。
