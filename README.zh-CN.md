# show-me-html

[English](README.md)

![写有 "Markdown is cheap. Show me HTML." 的横幅图：Markdown 文档展开成更丰富的 HTML artifact 仪表盘](assets/show-me-html-banner.png)

当普通聊天回答或 Markdown 文件不足以表达内容时，创建自包含的 HTML artifact。

`show-me-html` 是一个 Agent Skill，用于把密集内容转成更易读、更易导航、更可用的单文件 HTML：方案对比、技术解读、研究报告、评审记录、图解、数据看板、交互式工具、图表密集分析，以及类似演示文稿的页面。

这个 skill 设计得很轻量。主入口 `SKILL.md` 决定什么时候值得使用 HTML，然后只在任务需要时加载聚焦的参考文件。

## 安装

使用 `skills` CLI 安装：

```bash
npx skills add hi-zhenyu/show-me-html
```

然后描述你想要的 HTML artifact。通常不需要显式提到 skill 名称；支持自动激活 skill 的 agent 会在你要求 HTML、视觉布局、图表、图解或交互时自动触发它。

```text
Turn the solution comparison into a browsable HTML report.
```

如果你的 agent 没有自动触发，可以用 `$show-me-html` 显式调用。

如果你的 agent 支持本地 skill 目录，也可以手动安装（以 Codex 为例）：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R show-me-html "${CODEX_HOME:-$HOME/.codex}/skills/show-me-html"
```

手动安装后，如果 agent 没有自动发现新 skill，请重启或刷新 agent。

## 什么时候使用

当回答需要布局、层级、可视化、交互或分享时，使用 `show-me-html`：

- 多方案横向对比和决策矩阵
- 研究总结、技术解读和长报告
- 时间线、流程图、架构图、状态图和过程视图
- 带图表、公式、表格或 benchmark 的数据密集页面
- Prompt 调试器、任务分拣板、计算器、配置编辑器等一次性交互工具
- 对外分享的总结、PR 描述、评审记录、事故报告和演示型页面

如果只是短回答、简单命令列表、小 checklist、纯代码回复，或者需要长期手工 diff 的文件，继续使用 Markdown 更合适。

## 示例提示词

```text
Make a visual HTML comparison of these three implementation options.
```

```text
Turn this PDF summary into a technical deep-dive HTML page with diagrams and source notes.
```

```text
Build a small interactive HTML prompt tuner. It should export the final prompt as Markdown.
```

```text
Create a chart-heavy benchmark report as a single HTML file. Keep the source data inspectable.
```

## 重要说明

这个 skill 改善的是 agent 输出的形式、结构和可读性。它不会替代用户本身的任务要求。

例如，如果你要求 agent 解读一篇论文，这个 skill 可以把解读变成更清晰的 HTML artifact，包含分节、图解、图表、来源说明和导航。但解读的深度、严谨性、证据质量和实用性，仍然取决于你要求模型做什么。

使用这个 skill，是为了最大化模型呈现工作的效果，而不是替代你提出正确分析要求。

HTML artifact 的成本也高于 Markdown。它通常需要更多 agent 工作时间和更多 token，因此适合在更好的呈现、导航或交互值得额外成本时使用。

## 它会产出什么

默认情况下，这个 skill 会让 agent 生成一个自包含的 `.html` 文件：

- 内联 CSS 和 JavaScript，在可行时支持离线打开
- 语义化 HTML、响应式布局、键盘友好的控件和清晰的焦点状态
- 对事实性内容，把证据和来源说明放在关键结论附近
- 对实质性数据可视化，默认使用 ECharts
- 对复杂公式，使用 KaTeX 或 MathJax
- 对交互式工具，提供导出路径，让用户能把结果带到下一个工作流

安装这个 skill 不需要额外运行时依赖。生成的 artifact 可能会在合适时使用 ECharts、KaTeX 或 MathJax；当严格离线使用很重要时，skill 会要求 agent 提供 fallback 或明确依赖说明。

## 示例

本仓库包含同一份中文示例的 Markdown 和 HTML 版本，方便你对比线性文档和更丰富的 HTML artifact：

- Markdown：[`deepseek_v4_deep_dive_zh.md`](https://github.com/hi-zhenyu/show-me-html/blob/main/examples/deepseek_v4_deep_dive_zh.md)
- HTML：[`deepseek_v4_deep_dive_zh.html`](https://hi-zhenyu.github.io/show-me-html/examples/deepseek_v4_deep_dive_zh.html)

这些示例是产物展示，不是使用这个 skill 的必需输入。它们展示了推荐的结构：先给结论、分节导航、清晰视觉层级、来源说明，以及可本地打开的完整 HTML 文件。

## 仓库结构

```text
.github/workflows/pages.yml
.nojekyll
assets/
show-me-html/
  SKILL.md
  references/
examples/
README.md
README.zh-CN.md
LICENSE
.gitignore
```

- `show-me-html/SKILL.md` 是 skill 入口和触发描述。
- `show-me-html/references/` 存放 agent 按需读取的聚焦指导。
- `assets/` 存放 README 和项目媒体资源。
- `examples/` 存放 Markdown 和 HTML 示例产物。
- `.github/workflows/pages.yml` 会把实时 HTML 示例发布到 GitHub Pages。
- `.nojekyll` 用于避免 GitHub Pages 在发布静态文件时进行 Jekyll 处理。

## 致谢

这个 skill 受到 Thariq Shihipar 的文章 [The Unreasonable Effectiveness of HTML](https://thariqs.github.io/html-effectiveness/) 启发，尤其是其中关于 Markdown 在 agent 越来越强大后可能变成 “a restricting format” 的观察。

这个 skill 的 prompt design 也参考了 [`dogum/html-artifacts`](https://github.com/dogum/html-artifacts) skill 中的一些思路。

## 许可证

MIT。见 [`LICENSE`](LICENSE)。
