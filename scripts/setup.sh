#!/bin/bash
# exam-intelligence 环境一键安装脚本

set -e

echo "=== 考研智库环境安装 ==="

# 1. 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 2. 安装 pip 依赖
echo "安装 Python 依赖..."
pip install -r requirements.txt

# 3. 检查 Ollama
if ! command -v ollama &> /dev/null; then
    echo "Ollama 未安装，正在引导安装..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama 已安装: $(ollama --version)"
fi

# 4. 拉取模型
echo "拉取 Qwen3 模型（可能需要几分钟）..."
ollama pull qwen3:8b

echo "=== 安装完成 ==="
echo "运行以下命令启动："
echo "  streamlit run src/ui/main.py"