# IDA Pro MCP 智能逆向分析平台（增强版）

> 🚀 全自动 · 智能化 · 一站式逆向分析工具链
> 
> **本项目基于 [mrexodia/ida-pro-mcp](https://github.com/mrexodia/ida-pro-mcp) 二次开发增强，保留原核心功能并自行diy扩展了一些。**

---

## 🌟 项目简介

IDA Pro MCP 是一款集成 LLM（大语言模型）与 IDA Pro 的智能逆向分析平台，支持多客户端（Cursor、Cline、Roo Code、Windsurf、LM Studio等），一键自动完成函数/变量重命名、类型修复、注释、结构体声明、混淆检测、反调试检测、算法识别、爆破脚本生成与执行、主流程图自动生成、结构化分析报告导出等高阶功能。

> **本增强版在原版基础上，重点升级了混淆检测、基础算法识别、结构化报告等自动化能力，极大提升了逆向分析的智能化和自动化水平。**

适用于 CTF、二进制安全分析、恶意代码分析、自动化审计、团队协作等多种场景。

---

## ✨ 主要功能

- **一键自动分析**：只需一句 prompt，自动完成函数/变量重命名、类型修复、注释、结构体声明等繁琐工作。
- **多客户端支持**：兼容 Cline、Roo Code、Cursor、Windsurf、LM Studio 等主流 LLM 客户端。
- **批量处理与自动化**：支持批量重命名、批量注释、批量类型修复、批量 patch 点定位，极大提升分析效率。
- **混淆与反调试检测（增强）**：
  - 自动识别控制流平坦化、switch/case、间接跳转、死代码、字符串加密、反调试 API、TLS、PEB、int 0x2d 等多种混淆与反调试手段。
  - 检测结果更细致，支持报告 flattening、string_encryption、anti_debug、dead_code 等多维度信息。
- **基础算法自动识别（增强）**：
  - 支持自动检测 RC4、Base64、TEA、MD5、AES、DES、SHA1、SHA256、CRC32、XOR、BASE32/58/85、ROT13、Zlib、LZMA、Mersenne Twister 等常见加密/编码/哈希算法。
  - 检测结果包含算法类型和置信度，自动汇总进分析报告。
- **结构化分析报告（增强）**：
  - 一键导出 Markdown 格式的多维度分析报告（report.md），涵盖入口点、关键函数、反调试点、算法、混淆、主流程图、代码复杂度、交叉引用热点等。
  - 报告中详细展示每个函数的算法类型、置信度、混淆特征等，便于溯源和自动化分析。
- **爆破与动态分析辅助**：一键生成并可自动运行 angr/frida 脚本，支持符号执行、动态 hook、调试寄存器/内存自动字符串收集。
- **可视化主流程图**：自动生成 mermaid/graphviz 流程图，直观展示主线逻辑。
- **增量变更追踪**：自动记录所有重命名、注释、类型修复等变更，便于回溯和增量分析。
- **全中文注释与文档**：代码、接口、报告均为中文，极易上手和二次开发。

---

## 🆚 与原版 [mrexodia/ida-pro-mcp](https://github.com/mrexodia/ida-pro-mcp) 主要区别

- **混淆检测能力大幅增强**：不仅检测控制流平坦化，还支持 switch/case、间接跳转、死代码、字符串加密、反调试等多维度混淆特征。
- **基础算法识别更丰富**：支持十余种常见加密/编码/哈希算法的自动识别，结果更直观、置信度可量化。
- **结构化报告内容更全面**：每个函数的算法类型、混淆特征、置信度等均自动汇总进报告，便于团队协作和自动化溯源。
- **接口注释与文档更完善**：所有接口均有详细中文注释，便于二次开发和自动生成 API 文档。
- **兼容原版全部核心功能，且持续扩展新特性。**

---

## 🛠️ 安装与环境要求

### 1. 环境依赖
- **Python 3.11 及以上**
- **IDA Pro 7.x 及以上**（需支持 Python 3.11 插件）
- **依赖包**：`mcp>=1.6.0`（已在 pyproject.toml 中声明）

### 2. 安装步骤
（记得使用ida自带的python哦）
#### （1）克隆本项目
```bash
git clone https://github.com/namename333/idapromcp_333.git
cd ida-pro-mcp-main
```

#### （2）安装依赖
```bash
pip install -r requirements.txt  # 或使用 pyproject.toml/uv
```

#### （3）一键安装 MCP 服务器与 IDA 插件
```bash
python -m ida_pro_mcp.server --install
```
- 支持 Windows/Mac/Linux，自动检测并配置各主流 LLM 客户端（Cursor、Cline、Roo Code、Windsurf、LM Studio 等）
- 插件会自动复制到 IDA Pro 插件目录，无需手动操作

#### （4）启动 MCP 服务
- 启动 IDA Pro，加载目标二进制文件
- 在 IDA 菜单栏选择 `Edit -> Plugins -> MCP (Ctrl-Alt-M)` 启动服务
- 或命令行运行：
  ```bash
  python -m ida_pro_mcp.server
  ```

#### （5）在 LLM 客户端中配置/选择 MCP 服务器
- 参考自动生成的配置文件（如 `.cursor/mcp.json`、`cline_mcp_settings.json` 等）
- 也可手动配置，见下方“高级用法”

---

## 🚩 快速上手

1. **在支持 MCP 的 LLM 客户端（如 Cursor、Cline、Roo Code）中输入自然语言指令**，即可自动驱动所有分析流程。
2. **一键生成 report.md**，快速交付高质量分析报告。
3. **支持批量重命名、批量注释、自动类型修复、爆破脚本生成与执行等自动化操作。**

---

## 🔥 典型工作流

1. 启动 IDA Pro，加载目标二进制，启动 MCP 插件
2. 在 LLM 客户端输入分析指令（如“分析主流程”、“检测混淆”、“批量重命名函数”等）
3. 自动完成分析、注释、重命名、类型修复、爆破脚本生成等
4. 一键导出结构化分析报告（Markdown 格式，含主流程图、关键函数、反调试点等）
5. 支持增量变更追踪、自动回溯

---

## 📋 Prompt 模板

### 1. 函数分析模板
```text
请分析函数{function_name}的功能，包括：
1. 输入参数和返回值分析
2. 主要控制流程说明
3. 关键算法识别（如加密/解密、哈希、压缩等）
4. 潜在漏洞点或反调试手段
5. 生成易读的伪代码
```

### 2. 代码注释模板
```text
为以下汇编代码生成详细注释：
{assembly_code}

注释要求：
1. 每条指令的功能解释
2. 寄存器用途说明
3. 数据结构分析
4. 程序流程说明
```

### 3. 批量重命名模板
```text
对以下函数进行批量重命名：
{function_list}

命名规则：
1. 基于功能命名（如：encrypt_data、parse_header）
2. 使用下划线命名法
3. 避免过于冗长（控制在30字符以内）
```
### 4. 人话
```text
尝试加入了一些去混淆，和现有脚本，判断加密，生成文档等。
后续有机会再持续更新
```
---

## ⚙️ 高级用法

- **命令行参数**：
  - `--install` / `--uninstall`：一键安装/卸载 MCP 服务器与插件
  - `--transport stdio|http://host:port`：指定通信协议
  - `--unsafe`：启用不安全函数（如调试器操作等，需谨慎）
  - `--config`：生成 MCP 配置 JSON
- **自动生成的配置文件**：支持多客户端自动配置，详见 `src/ida_pro_mcp/server.py`
- **自定义开发/二次开发**：代码全中文注释，结构清晰，便于扩展

---

## 🤝 贡献与交流

- 欢迎 issue、PR、讨论区交流新功能建议、bug 反馈、自动化脚本分享等
- 本项目持续维护，欢迎 star & fork！
- 让逆向分析更高效、更智能、更自动化！

---

## 📝 开源协议与致谢

- 本项目基于 [mrexodia/ida-pro-mcp](https://github.com/mrexodia/ida-pro-mcp) 二次开发，遵循 MIT License
- 感谢原作者 Duncan Ogilvie 及社区贡献者
- 如需引用、二次开发、商用等请遵循 LICENSE 文件条款

---

如需更详细的功能演示、API文档、使用教程等，也可随时 issue/讨论区告知！
