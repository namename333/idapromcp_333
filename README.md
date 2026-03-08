# IDA Pro MCP 插件

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![IDA Pro Version](https://img.shields.io/badge/IDA%20Pro-7.5%2B-green.svg)](https://www.hex-rays.com/products/ida/)

## 1. 项目概述

IDA Pro MCP (Modding and Code Analysis Plugin) 是一个功能强大的 IDA Pro 插件，旨在提供丰富的代码分析、自动化和修改功能，帮助逆向工程师和安全研究人员更高效地进行二进制分析。该插件实现了标准 MCP 协议，支持与各种 MCP 客户端进行交互。本项目基于 [mrexodia/ida-pro-mcp](https://github.com/mrexodia/ida-pro-mcp) 进行二次开发与功能扩展。

### 1.1 核心价值

- **自动化分析流程**：减少重复工作，提高分析效率
- **增强的函数和变量分析**：提供更深入的代码洞察
- **批量处理功能**：支持大规模代码修改和分析
- **动态分析辅助**：生成 Frida 和 Angr 脚本，辅助动态分析
- **混淆检测**：识别常见的代码混淆技术
- **算法识别**：自动识别常见加密算法和哈希函数
- **标准 MCP 协议兼容**：支持与各种 MCP 客户端交互

### 1.2 技术栈

| 技术/依赖   | 版本要求 | 用途             |
| ----------- | -------- | ---------------- |
| Python      | >= 3.8   | 插件开发语言     |
| IDA Pro     | >= 7.5   | 宿主环境         |
| mcp         | >= 1.6.0 | MCP 协议支持     |
| pydantic    | >= 2.0.0 | 数据验证和序列化 |
| fastmcp     | 最新版   | MCP 服务器实现   |
| frida-tools | 可选     | 动态分析支持     |
| angr        | 可选     | 符号执行支持     |

## 2. 目录结构

```
ida-pro-mcp/
├── src/
│   └── ida_pro_mcp/
│       ├── __init__.py              # 插件入口
│       ├── idalib_server.py         # IDA 库服务器
│       ├── mcp-plugin.py            # 主插件文件
│       ├── mcp_config.json          # 插件配置
│       ├── script_utils.py          # 脚本生成辅助模块
│       ├── server.py                # MCP 服务器
│       └── server_generated.py      # 自动生成的服务器代码
├── .gitignore                       # Git 忽略配置
├── CONTRIBUTING.md                  # 贡献指南
├── README.md                        # 项目说明文档
├── ida-plugin.json                  # IDA 插件配置
├── pyproject.toml                   # 项目配置文件
├── requirements.txt                 # 依赖列表
├── test_script_utils.py             # 测试脚本
└── uv.lock                          # 依赖锁定文件
```

## 3. 核心功能

### 3.1 批量处理功能

- **批量重命名函数**：根据命名规则或模式批量重命名函数
- **批量添加注释**：自动为函数和变量添加描述性注释
- **批量设置变量类型**：根据分析结果自动设置本地变量类型
- **批量设置函数原型**：统一设置函数的参数和返回类型

### 3.2 混淆检测功能

- **控制流平坦化检测**：识别控制流混淆
- **字符串加密检测**：检测字符串加密和动态解密模式
- **反调试检测**：识别常见的反调试技术
- **死代码检测**：发现无用的代码块

### 3.3 算法识别功能

自动识别常见加密算法、哈希函数和编码方式：

- **加密算法**：AES、DES、RC4、TEA、XOR 等
- **哈希函数**：MD5、SHA1、SHA256、CRC32 等
- **编码方式**：Base64、Base32、Base58、Base85、Rot13 等
- **压缩算法**：zlib、LZMA 等
- **随机数生成**：Mersenne Twister 等

### 3.4 动态分析辅助功能

- **生成 Angr 符号执行脚本**：辅助进行符号执行和爆破
- **生成 Frida 动态分析脚本**：支持函数 Hook、内存监控和字符串监控
- **脚本保存和运行**：便捷地保存和执行生成的分析脚本
- **Windows 平台兼容性优化**：特别优化了在 Windows 环境下的使用体验

### 3.5 调用图分析

生成函数调用关系图，支持可视化展示：

- 生成邻接表格式调用图
- 生成 Mermaid 格式调用图，支持可视化
- 可指定递归深度

### 3.6 script_utils 模块

提供脚本生成辅助函数库，包含：

- **地址检测**：验证十六进制地址格式
- **目标表达式生成**：根据目标类型生成 Frida 表达式
- **脚本内容构建**：生成标准化的脚本模板
- **环境管理**：根据应用类型生成脚本环境

## 4. 安装步骤

### 4.1 系统要求

- IDA Pro 7.5+（支持 Python 3.8+）
- Python 3.8 或更高版本
- 支持 Windows、macOS 和 Linux 操作系统
- 如果使用 `idalib_server`，需准备可用的 IDA 运行库环境（例如正确设置 `IDADIR`）

### 4.2 安装方法

#### 手动安装

1. 克隆或下载项目代码：

```bash
git clone https://github.com/namename333/idapromcp_333.git
cd idapromcp_333
```

2. 安装依赖：

```bash
pip install -e .
# 安装额外的动态分析工具（推荐）
pip install angr frida-tools
```

3. 安装 IDA 插件：

```bash
# Windows 示例
ida-pro-mcp --install
```

### 4.3 验证安装

1. 启动 IDA Pro
2. 通过以下方式启动 MCP 服务器：
   - 菜单：`Edit -> Plugins -> MCP`
   - 快捷键：`Ctrl-Alt-M`
3. 服务器启动后，将在端口 13338 上监听 JSON-RPC 请求（默认读取 `src/ida_pro_mcp/mcp_config.json`）
4. 使用以下命令测试连接：

```python
import requests
import json

url = "http://localhost:13338/mcp"
headers = {"Content-Type": "application/json"}

payload = {
    "jsonrpc": "2.0",
    "method": "check_connection",
    "params": [],
    "id": 1
}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

## 5. 使用指南

### 5.1 基本用法

MCP 插件通过 JSON-RPC 协议提供 API，您可以使用任何支持 HTTP 请求的工具或编程语言与之交互。以下是一个基本的请求示例：

```python
import requests
import json

url = "http://localhost:13338/jsonrpc"
headers = {"Content-Type": "application/json"}

payload = {
    "jsonrpc": "2.0",
    "method": "get_function",
    "params": ["0x401000"],
    "id": 1
}

response = requests.post(url, data=json.dumps(payload), headers=headers)
result = response.json()
print(result)
```

### 5.2 MCP 服务端命令行参数（当前版本）

`python -m ida_pro_mcp.server --help` 当前可用参数：

- `--install`：安装 MCP 服务器配置和 IDA 插件
- `--uninstall`：卸载 MCP 服务器配置和 IDA 插件
- `--transport`：MCP 传输协议（默认 `stdio`，也可用 `http://127.0.0.1:8744`）
- `--ida-rpc`：指定 IDA RPC 地址（默认 `http://127.0.0.1:13338`）
- `--unsafe`：启用不安全函数（危险）
- `--config`：输出 MCP 配置 JSON
- `--auto-run-ida <binary>`：自动启动 IDA 并加载指定二进制文件

另外，`idalib` 模式入口为：

```bash
python -m ida_pro_mcp.idalib_server <input_path> --host 127.0.0.1 --port 8746
```

### 5.3 脚本生成示例

#### 5.3.1 生成 Frida Hook 脚本

```python
# 生成函数 Hook 脚本
hook_script = ida_pro_mcp.generate_frida_script(
    target="main",            # 目标函数名
    script_type="hook",       # 脚本类型为 hook
    options={
        "app_type": "native",  # 应用类型为原生二进制
        "log_args": True,       # 记录函数参数
        "log_return": True,     # 记录返回值
        "detailed_log": False   # 简洁日志输出
    }
)
```

#### 5.3.2 生成内存监控脚本

```python
# 生成内存监控脚本
memory_script = ida_pro_mcp.generate_frida_script(
    target="0x12345678",      # 要监控的内存地址
    script_type="memory_dump", # 脚本类型为内存监控
    options={
        "size": 1024,           # 监控内存大小（字节）
        "interval": 100,        # 监控间隔（毫秒）
        "compare_before_after": True # 比较内存前后变化
    }
)
```

#### 5.3.3 生成字符串监控脚本

```python
# 生成字符串监控脚本
string_script = ida_pro_mcp.generate_frida_script(
    target="",                 # 不需要特定目标函数
    script_type="string_hook",  # 脚本类型为字符串监控
    options={
        "string_functions": True,     # Hook 常见字符串函数
        "address_monitor": "0x10000000", # 监控特定内存区域
        "capture_callstack": True    # 捕获调用栈
    }
)
```

## 6. API 文档

### 6.1 元数据和基本信息获取

#### `get_metadata()`

**功能**：获取当前 IDB 文件的元数据信息

**返回**：包含元数据的字典

```python
{
    "file_path": "",  # 文件路径
    "file_size": 0,   # 文件大小（字节）
    "arch": "",       # 架构信息
    "bits": 0,        # 位数（32/64）
    "compiler": ""    # 编译器信息（如果可识别）
}
```

### 6.2 函数和变量分析

#### `get_function(function_address)`

**功能**：获取函数的详细信息

**参数**：

- `function_address`: 函数地址（字符串格式）

**返回**：包含函数信息的字典

```python
{
    "name": "",              # 函数名称
    "address": "",           # 函数地址
    "size": 0,               # 函数大小（字节）
    "prototype": "",         # 函数原型
    "is_imported": False,    # 是否为导入函数
    "is_thunk": False,       # 是否为 thunk 函数
    "callers": [],           # 调用此函数的函数列表
    "callees": []            # 此函数调用的函数列表
}
```

#### `list_functions()`

**功能**：列出数据库中的所有函数

**返回**：函数信息列表（每个元素与 `get_function` 返回格式相同）

### 6.3 反汇编和反编译

#### `decompile_function(function_address)`

**功能**：反编译函数并获取伪代码

**参数**：

- `function_address`: 函数地址

**返回**：反编译得到的伪代码字符串

#### `disassemble_function(function_address)`

**功能**：获取函数的汇编代码信息

**参数**：

- `function_address`: 函数地址

**返回**：包含汇编代码信息的字典

```python
{
    "address": "",       # 函数地址
    "name": "",          # 函数名称
    "instructions": [     # 指令列表
        {
            "address": "",  # 指令地址
            "mnem": "",     # 指令助记符
            "op_str": "",   # 操作数字符串
            "bytes": ""     # 指令字节码
        }
        # ...更多指令
    ]
}
```

### 6.4 调用图分析

#### `get_function_call_graph(start_address, depth=3, mermaid=False)`

**功能**：生成函数调用图

**参数**：

- `start_address`: 起始函数地址
- `depth`: 递归深度（默认 3）
- `mermaid`: 是否返回 mermaid 格式的图表定义（默认 False）

**返回**：包含调用图信息的字典

```python
{
    "graph": {},     # 邻接表或 mermaid 字符串
    "nodes": [],     # 节点列表（函数地址）
    "edges": []      # 边列表（调用关系）
}
```

**实现原理**：

调用图分析功能通过深度优先搜索 (DFS) 遍历函数调用关系，生成函数间的调用关系图。

**工作流程**：

1. **解析起始地址**：将输入的地址字符串转换为整数
2. **初始化数据结构**：创建访问集合、节点集合和边集合
3. **深度优先遍历**：
   - 从起始函数开始，获取所有被调用的函数
   - 记录调用关系（边）和函数（节点）
   - 递归处理被调用的函数，直到达到指定深度或所有函数都被访问
4. **生成结果**：根据参数返回邻接表或 mermaid 格式的调用图

**核心实现代码**：

```python
visited = set()
edges = set()
nodes = set()

def dfs(addr, d):
    if d < 0 or addr in visited:
        return
    visited.add(addr)
    func = idaapi.get_func(parse_address(addr))
    if not func:
        return
    nodes.add(addr)
    for ref in idautils.CodeRefsFrom(func.start_ea, 1):
        callee_func = idaapi.get_func(ref)
        if callee_func:
            callee_addr = hex(callee_func.start_ea)
            edges.add((addr, callee_addr))
            dfs(callee_addr, d-1)

dfs(start_address, depth)
```

### 6.5 混淆检测

#### `detect_obfuscation(function_address)`

**功能**：检测常见混淆模式

**参数**：

- `function_address`: 要检测混淆的函数地址

**返回**：包含混淆检测结果的字典

```python
{
    "flattening": False,         # 是否存在控制流平坦化
    "string_encryption": False,  # 是否存在字符串加密
    "anti_debug": False,         # 是否存在反调试技术
    "dead_code": False,          # 是否存在死代码
    "details": ""                # 详细信息
}
```

**实现原理**：

混淆检测功能通过分析函数的控制流、代码结构和特征模式来识别常见的混淆技术。

**检测方法**：

- **控制流平坦化检测**：分析函数的控制流图，检测异常的分支结构
- **字符串加密检测**：识别动态解密字符串的模式和特征
- **反调试检测**：查找常见的反调试指令和API调用
- **死代码检测**：识别无法执行的代码块

**工作流程**：

1. **获取函数信息**：获取目标函数的基本信息和反编译代码
2. **控制流分析**：生成并分析函数的控制流图
3. **特征匹配**：匹配已知的混淆模式和特征
4. **统计分析**：分析代码复杂度、分支数量、指令分布等统计特征
5. **结果生成**：返回检测结果和详细信息

**核心实现逻辑**：

```python
# 分析控制流图
flowchart = ida_gdl.FlowChart(func)
branch_count = sum(len(list(block.succs())) for block in flowchart)

# 分析反编译代码
code = decompile_function(function_address)

# 检测控制流平坦化
flattening = branch_count > 100  # 示例阈值检测

# 检测字符串加密
string_encryption = "memcpy" in code and "xor" in code  # 示例特征匹配

# 检测反调试
anti_debug = any(kw in code for kw in ["IsDebuggerPresent", "CheckRemoteDebuggerPresent"])
```

### 6.6 算法识别

#### `get_algorithm_signature(function_address)`

**功能**：自动识别常见算法

**参数**：

- `function_address`: 要识别算法的函数地址

**返回**：包含算法识别结果的字典

```python
{
    "algorithm": "unknown",  # 识别出的算法名称（多个用逗号分隔）
    "confidence": 0.0         # 置信度（0.0-1.0）
}
```

**实现原理**：

算法识别功能通过分析函数的指令序列、常量特征和模式匹配来识别常见的加密算法、哈希函数和编码方式。

**识别方法**：

- **特征匹配**：匹配已知算法的指令序列和常量特征
- **算法签名**：使用预定义的算法签名进行匹配
- **统计分析**：分析函数的指令分布、操作数特征等

**工作流程**：

1. **获取函数信息**：获取目标函数的反编译代码和指令序列
2. **特征提取**：提取函数的指令特征、常量、操作数等
3. **模式匹配**：与预定义的算法模式进行匹配
4. **置信度计算**：计算识别结果的置信度
5. **结果生成**：返回识别结果和置信度

**核心实现逻辑**：

```python
# 预定义的算法特征
algorithm_patterns = {
    "MD5": ["0x67452301", "0xefcdab89", "0x98badcfe", "0x10325476"],
    "SHA1": ["0x67452301", "0xefcdab89", "0x98badcfe", "0x10325476", "0xc3d2e1f0"],
    "AES": ["sbox", "0x63", "0x7c", "0x77", "0x7b"],
    "RC4": ["swap", "xor", "key"],
    "Base64": ["A-Za-z0-9+/", "padding", "encode", "decode"]
}

# 匹配算法特征
max_confidence = 0.0
best_algorithm = "unknown"
for algo, patterns in algorithm_patterns.items():
    matched = 0
    total = len(patterns)
    for pattern in patterns:
        if pattern in code:
            matched += 1
    confidence = matched / total
    if confidence > max_confidence:
        max_confidence = confidence
        best_algorithm = algo
```

### 6.7 动态分析辅助

#### `generate_frida_script(target, script_type, options=None)`

**功能**：生成 Frida 动态分析脚本

**参数**：

- `target`: 目标函数名或地址
- `script_type`: 脚本类型（'hook', 'memory_dump', 'string_hook'）
- `options`: 可选配置参数

**返回**：生成的 JavaScript 脚本代码

#### `save_generated_script(script_content, script_type, file_name=None)`

**功能**：保存生成的脚本到文件

**参数**：

- `script_content`: 脚本内容
- `script_type`: 脚本类型（'angr' 或 'frida'）
- `file_name`: 保存的文件名（可选，默认生成带时间戳的文件名）

**返回**：保存结果信息，包含文件路径和运行命令提示

## 7. 典型工作流

### 7.1 基础分析流程

1. 加载二进制文件到 IDA Pro
2. 启动 MCP 插件服务器（Ctrl-Alt-M）
3. 使用 `get_metadata()` 获取基本信息
4. 使用 `list_functions()` 浏览所有函数
5. 使用 `get_function()` 深入分析特定函数
6. 使用 `decompile_function()` 获取伪代码
7. 使用 `detect_obfuscation()` 检测混淆
8. 使用 `get_algorithm_signature()` 识别算法

### 7.2 动态分析流程

1. 确定分析目标函数
2. 使用 `generate_frida_script()` 生成 Hook 脚本
3. 使用 `save_generated_script()` 保存脚本
4. 使用 `run_external_script()` 运行脚本并观察结果
5. 分析输出结果并调整脚本参数

### 7.3 批量处理流程

1. 选择需要处理的函数集
2. 使用 `batch_rename_functions()` 统一命名规范
3. 使用 `batch_set_local_variable_types()` 设置变量类型
4. 使用 `batch_set_function_prototypes()` 规范函数原型
5. 使用 `batch_add_comments()` 添加描述性注释

## 8. 高级用法

### 8.1 使用调用图进行分析

1. 使用 `get_function_call_graph()` 生成调用图
2. 分析关键函数的调用关系
3. 识别程序的核心组件和数据流
4. 使用 Mermaid 渲染调用图，可视化分析结果

### 8.2 调试器控制

MCP 插件提供了基本的调试器控制功能，可用于自动化调试过程：

- `dbg_continue_process()`: 继续执行进程
- `dbg_run_to(address)`: 运行到指定地址
- `dbg_set_breakpoint(address)`: 设置断点
- `dbg_remove_breakpoint(address)`: 移除断点

### 8.3 自定义脚本生成

通过扩展 `script_utils.py` 模块，您可以自定义脚本生成逻辑，满足特定的分析需求：

```python
# 示例：扩展 script_utils.py，添加自定义脚本生成函数
def _generate_custom_script(target, options):
    # 自定义脚本生成逻辑
    pass
```

## 9. 故障排除

### 9.1 常见问题

| 问题                  | 可能原因                            | 解决方案                                      |
| --------------------- | ----------------------------------- | --------------------------------------------- |
| 插件加载失败          | 插件目录结构不正确                  | 检查插件文件是否正确放置在 IDA Pro 插件目录中 |
| 服务器无法启动        | 端口 13338 已被占用                 | 修改 `src/ida_pro_mcp/mcp_config.json` 中的端口设置 |
| Python 版本不匹配     | IDA Pro 使用的 Python 版本低于 3.8  | 升级 IDA Pro 或使用兼容的 Python 版本         |
| 依赖错误              | 缺少必要的依赖包                    | 重新运行 `pip install -r requirements.txt`  |
| script_utils 模块错误 | 配置文件中 script_utils_path 不正确 | 检查 config.json 中的 script_utils_path 配置  |
| Frida 脚本生成失败    | 目标函数名或地址无效                | 确保提供的目标是有效的                        |

### 9.2 日志和调试

- MCP 插件会在 IDA 控制台输出 [MCP] 前缀的日志
- 检查这些日志可以帮助排查问题
- 可以通过修改 `server.py` 中的 log_level 来调整日志级别

## 10. 贡献指南

我们欢迎社区贡献！如果您想为 IDA Pro MCP 插件做出贡献，请参考以下指南：

### 10.1 开发环境设置

1. 克隆项目仓库
2. 安装开发依赖：`pip install -r requirements.txt`
3. 安装开发工具：`pip install pytest black isort`
4. 运行测试：`python test_script_utils.py`

### 10.2 代码规范

- 遵循 PEP 8 代码风格
- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 为所有函数添加文档字符串
- 编写单元测试

### 10.3 提交流程

1. Fork 项目仓库
2. 创建一个新的分支
3. 提交您的更改
4. 运行测试确保所有测试通过
5. 创建 Pull Request

### 10.4 报告问题

如果您发现任何问题或有建议，请在 GitHub Issues 中报告：

https://github.com/namename333/idapromcp_333/issues

## 11. 版本历史

### v1.4 (当前)

- 保留并增强原有批量分析、混淆检测、算法识别与脚本生成能力
- README 与当前实现对齐（端口、命令行参数、依赖版本）
- 默认推荐使用 `mcp_config.json` 统一管理端口配置

### v1.3

- 增强版功能沉淀版本，包含 script_utils、动态分析辅助与调用图能力

### v1.x

- 初始版本
- 基本 MCP 协议支持
- 核心功能实现

## 12. 许可证

本项目采用 MIT 许可证，详细信息请查看 [LICENSE](LICENSE) 文件。

## 13. 联系方式

### 开发者

- **name** - 开发者（QQ：1559820232）
- **grand** - 开发者 (QQ: 3527424707)
- **Britney** - 开发者 (QQ: 2855057900)

### 项目链接

- GitHub 仓库：https://github.com/namename333/idapromcp_333
- 原项目：https://github.com/mrexodia/ida-pro-mcp

## 14. 致谢

- 感谢 mrexodia 提供的原始 IDA Pro MCP 插件
- 感谢所有为该项目做出贡献的开发者和用户
