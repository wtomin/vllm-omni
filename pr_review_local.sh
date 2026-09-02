#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# 基本参数
# =========================

# 必须在 Git 仓库中执行
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"

if [[ -z "$ROOT" ]]; then
    echo "错误：当前目录不是 Git 仓库"
    exit 1
fi

cd "$ROOT"

# =========================
# 清除可能产生冲突的旧变量
# =========================

unset CONFIG__MODEL
unset CONFIG__FALLBACK_MODELS
unset PR_AGENT__CONFIG__MODEL
unset PR_AGENT__CONFIG__FALLBACK_MODELS
unset PR_AGENT_MODEL
unset PR_AGENT_CONFIG_MODEL


# =========================
# 模型配置
# =========================

# PR-Agent/Dynaconf 的 config 节配置
export CONFIG__MODEL="openai/deepseek-v4-flash"
export CONFIG__FALLBACK_MODELS='[]'
export CONFIG__CUSTOM_MODEL_MAX_TOKENS="131072"

# 同时设置 PR_AGENT 前缀
export PR_AGENT__CONFIG__MODEL="openai/deepseek-v4-flash"
export PR_AGENT__CONFIG__FALLBACK_MODELS='[]'
export PR_AGENT__CONFIG__CUSTOM_MODEL_MAX_TOKENS="131072"
# 可选：推理模型
# export CONFIG__MODEL="openai/deepseek-reasoner"
# export CONFIG__FALLBACK_MODELS='["openai/deepseek-reasoner"]'

# =========================
# 输出行为
# =========================

export CONFIG__PUBLISH_OUTPUT="false"
export CONFIG__VERBOSITY_LEVEL="2"

# =========================
# Git Provider
# =========================

export CONFIG__GIT_PROVIDER="github"
export CONFIG__PATCH_EXTRA_LINES="3"

# =========================
# PR 描述配置
# =========================

export PR_DESCRIPTION__EXTRA_INSTRUCTIONS='请使用中文生成 PR 描述。要求：
1. 给评审者一个清晰的心理模型，用平实的语言说明本次修改的整体架构意图
2. 按模块分类介绍：每个模块对应哪些文件的修改，功能变化是什么
3. 详细列出修改的公共接口（函数签名、类定义、API 端点等）
4. 分析潜在风险：兼容性破坏、性能影响、安全漏洞、并发问题等
5. 使用 Markdown 格式，结构清晰'

export PR_DESCRIPTION__PUBLISH_LABELS="false"
export PR_DESCRIPTION__ADD_ORIGINAL_USER_DESCRIPTION="true"
export PR_DESCRIPTION__USE_BULLET_POINTS="true"
export PR_DESCRIPTION__ENABLE_SEMANTIC_FILES_TYPES="true"
export PR_DESCRIPTION__COLLAPSIBLE_FILE_LIST="adaptive"
export PR_DESCRIPTION__INLINE_FILE_SUMMARY="false"
export PR_DESCRIPTION__ENABLE_LARGE_PR_HANDLING="true"

# =========================
# /ask 配置
# =========================

export PR_QUESTIONS__EXTRA_INSTRUCTIONS='Please ask questions in English only.
Focus on:
1. Areas where the author'\''s intent is unclear or ambiguous
2. Logical contradictions or inconsistencies in the implementation
3. Missing context that would help reviewers understand the change
4. Potential edge cases not addressed
5. Questions should be specific and reference line numbers where possible'

# =========================
# /improve 配置
# =========================

export PR_CODE_SUGGESTIONS__EXTRA_INSTRUCTIONS='Please provide all code suggestions in English only.
For each suggestion:
1. Include the exact file path and line number range
2. Explain the problem clearly
3. Provide a concrete code example showing the improved version
4. Rate the importance: [critical], [important], or [minor]
5. Focus on: code quality, maintainability, performance, security, and best practices
6. Take .claude/skills/ related skill (e.g., review-pr) as references'

export PR_CODE_SUGGESTIONS__NUM_CODE_SUGGESTIONS="8"
export PR_CODE_SUGGESTIONS__COMMITABLE_CODE_SUGGESTIONS="false"
export PR_CODE_SUGGESTIONS__RANK_SUGGESTIONS="true"
export PR_CODE_SUGGESTIONS__SELF_REFLECT_ON_SUGGESTIONS="true"
export PR_CODE_SUGGESTIONS__SUGGESTIONS_SCORE_THRESHOLD="0"

# =========================
# OpenAI 兼容 API
# =========================

: "${OPENAI_API_KEY:=}"

if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "错误：请先设置 OPENAI_API_KEY"
    echo
    echo '示例：'
    echo 'export OPENAI_API_KEY="sk-xxxxxxxx"'
    exit 1
fi

# PR-Agent 0.44.0 常用的 OpenAI 配置
export OPENAI__KEY="$OPENAI_API_KEY"
export OPENAI__API_BASE="${OPENAI_API_BASE:-https://api.deepseek.com/v1}"

# =========================
# GitHub 配置
# =========================

: "${GITHUB_TOKEN:=}"

if [[ -n "$GITHUB_TOKEN" ]]; then
    export GITHUB__USER_TOKEN="$GITHUB_TOKEN"
fi

# =========================
# 打印验证信息
# =========================

echo "========================================"
echo "PR-Agent 环境变量配置"
echo "========================================"
echo "仓库目录:       $ROOT"
echo "模型:           $CONFIG__MODEL"
echo "备用模型:       $CONFIG__FALLBACK_MODELS"
echo "发布输出:       $CONFIG__PUBLISH_OUTPUT"
echo "API 地址:       $OPENAI__API_BASE"
echo "OpenAI Key:     已设置"
if [[ -n "${GITHUB__USER_TOKEN:-}" ]]; then
    echo "GitHub Token:   已设置"
else
    echo "GitHub Token:   未设置"
fi
echo "========================================"


# ========== 配置区 ==========
PR_NUMBER=$1
REPO="vllm-project/vllm-omni"  # 修改为你的仓库
OUTPUT_DIR="tmp/pr-reviews"

# 检查参数
if [ -z "$PR_NUMBER" ]; then
    echo "Usage: ./pr_review_local.sh <PR_NUMBER>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "��� 正在获取 PR #$PR_NUMBER 的 diff..."

# ========== 步骤 1: 获取 PR diff ==========
# 方法 A: 使用 gh CLI 拉取 PR diff（推荐）
gh pr diff "$PR_NUMBER" --repo "$REPO" > "$OUTPUT_DIR/pr-${PR_NUMBER}.diff"

# 方法 B: 如果你已经在仓库目录中，也可以直接用 git
# git fetch origin pull/${PR_NUMBER}/head:pr-${PR_NUMBER}
# git diff main...pr-${PR_NUMBER} > "$OUTPUT_DIR/pr-${PR_NUMBER}.diff"

echo "✅ Diff 已保存到 $OUTPUT_DIR/pr-${PR_NUMBER}.diff"

# ========== 步骤 2: 生成中文 PR 描述 (/describe) ==========
echo ""
echo "��� 正在生成中文 PR 描述 (describe)..."
python -m pr_agent.cli \
    --diff-file "$OUTPUT_DIR/pr-${PR_NUMBER}.diff" \
    --output "$OUTPUT_DIR/pr-${PR_NUMBER}-describe.md" \
    describe

echo "✅ PR 描述已保存到 $OUTPUT_DIR/pr-${PR_NUMBER}-describe.md"

# ========== 步骤 3: 生成英文提问 (/ask) ==========
echo ""
echo "❓ 正在生成英文提问 (ask)..."
python -m pr_agent.cli \
    --diff-file "$OUTPUT_DIR/pr-${PR_NUMBER}.diff" \
    --output "$OUTPUT_DIR/pr-${PR_NUMBER}-ask.md" \
    ask "Please analyze this PR and ask specific questions about unclear logic, contradictions, or missing context. Reference specific line numbers in your questions."

echo "✅ 提问已保存到 $OUTPUT_DIR/pr-${PR_NUMBER}-ask.md"

# ========== 步骤 4: 生成英文改进建议 (/improve) ==========
echo ""
echo "��� 正在生成英文改进建议 (improve)..."
python -m pr_agent.cli \
    --diff-file "$OUTPUT_DIR/pr-${PR_NUMBER}.diff" \
    --output "$OUTPUT_DIR/pr-${PR_NUMBER}-improve.md" \
    improve

echo "✅ 改进建议已保存到 $OUTPUT_DIR/pr-${PR_NUMBER}-improve.md"

# ========== 完成 ==========
echo ""
echo "��� 全部完成！输出文件："
echo "  - PR 描述（中文）: $OUTPUT_DIR/pr-${PR_NUMBER}-describe.md"
echo "  - 提问（英文）   : $OUTPUT_DIR/pr-${PR_NUMBER}-ask.md"
echo "  - 改进建议（英文）: $OUTPUT_DIR/pr-${PR_NUMBER}-improve.md"
