# NOTE: This file has been automatically generated, do not modify!
# Architecture based on https://github.com/namename333/idapromcp_333 (MIT License)
import sys
if sys.version_info >= (3, 12):
    from typing import Annotated, Optional, TypedDict, Generic, TypeVar, NotRequired, Union
else:
    from typing_extensions import Annotated, Optional, TypedDict, Generic, TypeVar, NotRequired
    from typing import Union
from pydantic import Field

T = TypeVar("T")

class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str

class Function(TypedDict):
    address: str
    name: str
    size: str

class ConvertedNumber(TypedDict):
    decimal: str
    hexadecimal: str
    bytes: str
    ascii: Optional[str]
    binary: str

class Page(TypedDict, Generic[T]):
    data: list[T]
    next_offset: Optional[int]

class Global(TypedDict):
    address: str
    name: str

class Import(TypedDict):
    address: str
    imported_name: str
    module: str

class String(TypedDict):
    address: str
    length: int
    string: str

class DisassemblyLine(TypedDict):
    segment: NotRequired[str]
    address: str
    label: NotRequired[str]
    instruction: str
    comments: NotRequired[list[str]]

class Argument(TypedDict):
    name: str
    type: str

class DisassemblyFunction(TypedDict):
    name: str
    start_ea: str
    return_type: NotRequired[str]
    arguments: NotRequired[list[Argument]]
    stack_frame: list[dict]
    lines: list[DisassemblyLine]

class Xref(TypedDict):
    address: str
    type: str
    function: Optional[Function]

class StackFrameVariable(TypedDict):
    name: str
    offset: str
    size: str
    type: str

class StructureMember(TypedDict):
    name: str
    offset: str
    size: str
    type: str

class StructureDefinition(TypedDict):
    name: str
    size: str
    members: list[StructureMember]

@mcp.tool()
def check_connection() -> dict:
    """
    标准MCP协议接口：检查与服务器的连接
    用于客户端验证服务是否正常运行
    """
    return make_jsonrpc_request('check_connection')

@mcp.tool()
def get_methods() -> list[dict]:
    """
    获取所有可用的JSON-RPC方法列表及其元数据
    支持MCP协议的自描述功能
    """
    return make_jsonrpc_request('get_methods')

@mcp.tool()
def get_metadata() -> Metadata:
    """获取当前 IDB 的元数据"""
    return make_jsonrpc_request('get_metadata')

@mcp.tool()
def get_function_by_name(name: Annotated[str, Field(description='要获取的函数名称')]) -> Function:
    """根据函数名称获取函数"""
    return make_jsonrpc_request('get_function_by_name', name)

@mcp.tool()
def get_function_by_address(address: Annotated[str, Field(description='要获取的函数地址')]) -> Function:
    """根据函数地址获取函数"""
    return make_jsonrpc_request('get_function_by_address', address)

@mcp.tool()
def get_current_address() -> str:
    """获取用户当前选中的地址"""
    return make_jsonrpc_request('get_current_address')

@mcp.tool()
def get_current_function() -> Optional[Function]:
    """获取用户当前选中的函数"""
    return make_jsonrpc_request('get_current_function')

@mcp.tool()
def convert_number(text: Annotated[str, Field(description='要转换的数字的文本表示')], size: Annotated[Optional[int], Field(description='变量的大小（字节）')]) -> ConvertedNumber:
    """将数字（十进制、十六进制）转换为不同表示"""
    return make_jsonrpc_request('convert_number', text, size)

@mcp.tool()
def list_functions(offset: Annotated[int, Field(description='从 (0) 开始列出偏移量')], count: Annotated[int, Field(description='要列出的函数数量 (100 是默认值，0 表示剩余)')]) -> Page[Function]:
    """列出数据库中的所有函数（分页）
    
    警告: 此API已弃用，请使用更通用的query_database API
    """
    return make_jsonrpc_request('list_functions', offset, count)

@mcp.tool()
def query_database(entity_type: Annotated[str, Field(description="实体类型: 'functions', 'globals', 'strings', 'imports'")], offset: Annotated[int, Field(description='从 (0) 开始列出偏移量')], count: Annotated[int, Field(description='要列出的实体数量 (100 是默认值，0 表示剩余)')], filter: Annotated[str, Field(description='可选的过滤模式，空字符串表示无过滤')]) -> Page[dict]:
    """统一的数据库查询API，可查询各种实体类型
    
    用于替代特定的list_xxx和list_xxx_filter函数
    """
    return make_jsonrpc_request('query_database', entity_type, offset, count, filter)

@mcp.tool()
def list_globals_filter(offset: Annotated[int, Field(description='从 (0) 开始列出偏移量')], count: Annotated[int, Field(description='要列出的全局变量数量 (100 是默认值，0 表示剩余)')], filter: Annotated[str, Field(description='要应用的过滤器 (必需参数，空字符串表示无过滤). 大小写不敏感包含或 /regex/ 语法')]) -> Page[Global]:
    """列出数据库中的匹配全局变量（分页，过滤）"""
    return make_jsonrpc_request('list_globals_filter', offset, count, filter)

@mcp.tool()
def list_globals(offset: Annotated[int, Field(description='从 (0) 开始列出偏移量')], count: Annotated[int, Field(description='要列出的全局变量数量 (100 是默认值，0 表示剩余)')]) -> Page[Global]:
    """列出数据库中的所有全局变量（分页）
    
    警告: 此API已弃用，请使用更通用的query_database API
    """
    return make_jsonrpc_request('list_globals', offset, count)

@mcp.tool()
def list_imports(offset: Annotated[int, Field(description='从 (0) 开始列出偏移量')], count: Annotated[int, Field(description='要列出的导入符号数量 (100 是默认值，0 表示剩余)')]) -> Page[Import]:
    """ 列出所有导入符号及其名称和模块（分页） """
    return make_jsonrpc_request('list_imports', offset, count)

@mcp.tool()
def list_strings_filter(offset: Annotated[int, Field(description='从 (0) 开始列出偏移量')], count: Annotated[int, Field(description='要列出的字符串数量 (100 是默认值，0 表示剩余)')], filter: Annotated[str, Field(description='要应用的过滤器 (必需参数，空字符串表示无过滤). 大小写不敏感包含或 /regex/ 语法')]) -> Page[String]:
    """列出数据库中的匹配字符串（分页，过滤）"""
    return make_jsonrpc_request('list_strings_filter', offset, count, filter)

@mcp.tool()
def list_strings(offset: Annotated[int, Field(description='从 (0) 开始列出偏移量')], count: Annotated[int, Field(description='要列出的字符串数量 (100 是默认值，0 表示剩余)')]) -> Page[String]:
    """列出数据库中的所有字符串（分页）
    
    警告: 此API已弃用，请使用更通用的query_database API
    """
    return make_jsonrpc_request('list_strings', offset, count)

@mcp.tool()
def list_local_types():
    """列出数据库中的所有本地类型"""
    return make_jsonrpc_request('list_local_types')

@mcp.tool()
def decompile_function(address: Annotated[str, Field(description='要反编译的函数地址')]) -> str:
    """反编译给定地址的函数"""
    return make_jsonrpc_request('decompile_function', address)

@mcp.tool()
def disassemble_function(start_address: Annotated[str, Field(description='要反汇编的函数地址')]) -> DisassemblyFunction:
    """获取函数汇编代码"""
    return make_jsonrpc_request('disassemble_function', start_address)

@mcp.tool()
def get_xrefs_to(address: Annotated[str, Field(description='要获取交叉引用的地址')]) -> list[Xref]:
    """获取给定地址的所有交叉引用"""
    return make_jsonrpc_request('get_xrefs_to', address)

@mcp.tool()
def get_xrefs_to_field(struct_name: Annotated[str, Field(description='结构体名称 (类型)')], field_name: Annotated[str, Field(description='要获取交叉引用的字段名称 (成员)')]) -> list[Xref]:
    """获取命名结构体字段 (成员) 的所有交叉引用"""
    return make_jsonrpc_request('get_xrefs_to_field', struct_name, field_name)

@mcp.tool()
def get_entry_points() -> list[Function]:
    """获取数据库中的所有入口点"""
    return make_jsonrpc_request('get_entry_points')

@mcp.tool()
def set_comment(address: Annotated[str, Field(description='要设置注释的函数地址')], comment: Annotated[str, Field(description='注释文本')]):
    """设置给定函数反汇编和伪代码中的注释"""
    return make_jsonrpc_request('set_comment', address, comment)

@mcp.tool()
def rename_local_variable(function_address: Annotated[str, Field(description='包含变量的函数地址')], old_name: Annotated[str, Field(description='变量的当前名称')], new_name: Annotated[str, Field(description='变量的新名称 (空表示默认名称)')]):
    """重命名函数中的本地变量"""
    return make_jsonrpc_request('rename_local_variable', function_address, old_name, new_name)

@mcp.tool()
def rename_global_variable(old_name: Annotated[str, Field(description='全局变量的当前名称')], new_name: Annotated[str, Field(description='全局变量的新名称 (空表示默认名称)')]):
    """重命名全局变量"""
    return make_jsonrpc_request('rename_global_variable', old_name, new_name)

@mcp.tool()
def set_global_variable_type(variable_name: Annotated[str, Field(description='全局变量的名称')], new_type: Annotated[str, Field(description='变量的新类型')]):
    """设置全局变量的类型"""
    return make_jsonrpc_request('set_global_variable_type', variable_name, new_type)

@mcp.tool()
def get_global_variable_value_by_name(variable_name: Annotated[str, Field(description='全局变量的名称')]) -> str:
    """
    读取全局变量的值（如果编译时已知）

    优先使用此函数，而不是 `data_read_*` 函数。
    """
    return make_jsonrpc_request('get_global_variable_value_by_name', variable_name)

@mcp.tool()
def get_global_variable_value_at_address(ea: Annotated[str, Field(description='全局变量的地址')]) -> str:
    """
    通过地址读取全局变量的值（如果编译时已知）

    优先使用此函数，而不是 `data_read_*` 函数。
    """
    return make_jsonrpc_request('get_global_variable_value_at_address', ea)

@mcp.tool()
def rename_function(function_address: Annotated[str, Field(description='要重命名的函数地址')], new_name: Annotated[str, Field(description='函数的新名称 (空表示默认名称)')]):
    """重命名函数"""
    return make_jsonrpc_request('rename_function', function_address, new_name)

@mcp.tool()
def set_function_prototype(function_address: Annotated[str, Field(description='函数地址')], prototype: Annotated[str, Field(description='新的函数原型')]):
    """设置函数原型"""
    return make_jsonrpc_request('set_function_prototype', function_address, prototype)

@mcp.tool()
def declare_c_type(c_declaration: Annotated[str, Field(description='类型C声明。示例包括：typedef int foo_t; struct bar { int a; bool b; };')]):
    """从C声明创建或更新本地类型"""
    return make_jsonrpc_request('declare_c_type', c_declaration)

@mcp.tool()
def set_local_variable_type(function_address: Annotated[str, Field(description='要反编译的函数地址')], variable_name: Annotated[str, Field(description='变量名称')], new_type: Annotated[str, Field(description='变量的新类型')]):
    """设置本地变量的类型"""
    return make_jsonrpc_request('set_local_variable_type', function_address, variable_name, new_type)

@mcp.tool()
def get_stack_frame_variables(function_address: Annotated[str, Field(description='要获取栈帧变量的反汇编函数地址')]) -> list[StackFrameVariable]:
    """获取给定函数的栈帧变量"""
    return make_jsonrpc_request('get_stack_frame_variables', function_address)

@mcp.tool()
def get_defined_structures() -> list[StructureDefinition]:
    """返回所有定义的结构体列表"""
    return make_jsonrpc_request('get_defined_structures')

@mcp.tool()
def rename_stack_frame_variable(function_address: Annotated[str, Field(description='要设置栈帧变量的反汇编函数地址')], old_name: Annotated[str, Field(description='变量的当前名称')], new_name: Annotated[str, Field(description='变量的新名称 (空表示默认名称)')]):
    """更改IDA函数中栈帧变量的名称"""
    return make_jsonrpc_request('rename_stack_frame_variable', function_address, old_name, new_name)

@mcp.tool()
def create_stack_frame_variable(function_address: Annotated[str, Field(description='要设置栈帧变量的反汇编函数地址')], offset: Annotated[str, Field(description='栈帧变量的偏移量')], variable_name: Annotated[str, Field(description='栈变量名称')], type_name: Annotated[str, Field(description='栈变量类型')]):
    """对于给定的函数，在指定偏移量处创建一个栈变量并设置特定类型"""
    return make_jsonrpc_request('create_stack_frame_variable', function_address, offset, variable_name, type_name)

@mcp.tool()
def set_stack_frame_variable_type(function_address: Annotated[str, Field(description='要设置栈帧变量的反汇编函数地址')], variable_name: Annotated[str, Field(description='栈变量名称')], type_name: Annotated[str, Field(description='栈变量类型')]):
    """对于给定的反汇编函数，设置栈变量的类型"""
    return make_jsonrpc_request('set_stack_frame_variable_type', function_address, variable_name, type_name)

@mcp.tool()
def delete_stack_frame_variable(function_address: Annotated[str, Field(description='要设置栈帧变量的函数地址')], variable_name: Annotated[str, Field(description='栈变量名称')]):
    """删除给定函数的命名栈变量"""
    return make_jsonrpc_request('delete_stack_frame_variable', function_address, variable_name)

@mcp.tool()
def read_memory_bytes(memory_address: Annotated[str, Field(description='要读取的字节地址')], size: Annotated[int, Field(description='要读取的内存大小')]) -> str:
    """
    读取指定地址的字节。

    仅当 `get_global_variable_at` 和 `get_global_variable_by_name`
    都失败时才使用此函数。
    """
    return make_jsonrpc_request('read_memory_bytes', memory_address, size)

@mcp.tool()
def dbg_get_registers() -> list[dict[str, str]]:
    """获取所有寄存器及其值。此函数仅在调试时可用。"""
    return make_jsonrpc_request('dbg_get_registers')

@mcp.tool()
def dbg_get_call_stack() -> list[dict[str, str]]:
    """获取当前调用堆栈。"""
    return make_jsonrpc_request('dbg_get_call_stack')

@mcp.tool()
def dbg_control_process(action: Annotated[str, Field(description="调试操作类型: 'start', 'exit', 'continue', 'run_to'")], address: Annotated[Optional[str], Field(description='目标地址，仅run_to操作需要')]=None) -> str:
    """统一的调试器控制接口
    
    Args:
        action: 调试操作类型，支持'start', 'exit', 'continue', 'run_to'
        address: 目标地址，仅在action为'run_to'时需要
    
    Returns:
        操作结果消息
    """
    return make_jsonrpc_request('dbg_control_process', action, address)

@mcp.tool()
def dbg_start_process() -> str:
    """启动调试器 (已弃用，请使用dbg_control_process)"""
    return make_jsonrpc_request('dbg_start_process')

@mcp.tool()
def dbg_exit_process() -> str:
    """退出调试器 (已弃用，请使用dbg_control_process)"""
    return make_jsonrpc_request('dbg_exit_process')

@mcp.tool()
def dbg_continue_process() -> str:
    """继续调试器 (已弃用，请使用dbg_control_process)"""
    return make_jsonrpc_request('dbg_continue_process')

@mcp.tool()
def dbg_run_to(address: Annotated[str, Field(description='运行调试器到指定地址')]) -> str:
    """运行调试器到指定地址 (已弃用，请使用dbg_control_process)"""
    return make_jsonrpc_request('dbg_run_to', address)

@mcp.tool()
def dbg_manage_breakpoint(action: Annotated[str, Field(description="断点操作类型: 'list', 'set', 'delete', 'enable'")], address: Annotated[Optional[str], Field(description='断点地址，仅set/delete/enable操作需要')]=None, enable: Annotated[Optional[bool], Field(description='是否启用断点，仅enable操作需要')]=None) -> Union[str, list[dict[str, str]]]:
    """统一的断点管理接口
    
    Args:
        action: 断点操作类型，支持'list', 'set', 'delete', 'enable'
        address: 断点地址，仅在action为'set', 'delete', 'enable'时需要
        enable: 是否启用断点，仅在action为'enable'时需要
    
    Returns:
        操作结果消息或断点列表
    """
    return make_jsonrpc_request('dbg_manage_breakpoint', action, address, enable)

@mcp.tool()
def dbg_list_breakpoints():
    """列出程序中的所有断点 (已弃用，请使用dbg_manage_breakpoint)"""
    return make_jsonrpc_request('dbg_list_breakpoints')

@mcp.tool()
def dbg_set_breakpoint(address: Annotated[str, Field(description='在指定地址设置断点')]) -> str:
    """在指定地址设置断点 (已弃用，请使用dbg_manage_breakpoint)"""
    return make_jsonrpc_request('dbg_set_breakpoint', address)

@mcp.tool()
def dbg_delete_breakpoint(address: Annotated[str, Field(description='del a breakpoint at the specified address')]) -> str:
    """del a breakpoint at the specified address (已弃用，请使用dbg_manage_breakpoint)"""
    return make_jsonrpc_request('dbg_delete_breakpoint', address)

@mcp.tool()
def dbg_enable_breakpoint(address: Annotated[str, Field(description='Enable or disable a breakpoint at the specified address')], enable: Annotated[bool, Field(description='Enable or disable a breakpoint')]) -> str:
    """Enable or disable a breakpoint at the specified address (已弃用，请使用dbg_manage_breakpoint)"""
    return make_jsonrpc_request('dbg_enable_breakpoint', address, enable)

@mcp.tool()
def generate_angr_script(function_address: Annotated[str, Field(description='要分析的函数地址')], script_type: Annotated[str, Field(description="脚本类型: 'symbolic_execution', 'brute_force', 'control_flow'")], options: Annotated[dict, Field(description='可选配置参数，例如输入约束、输出格式等')]=None) -> str:
    """
    生成angr符号执行或爆破脚本。
    参数：
        function_address: 目标函数地址
        script_type: 脚本类型
        options: 可选配置
    返回：
        生成的Python脚本代码
    """
    return make_jsonrpc_request('generate_angr_script', function_address, script_type, options)

@mcp.tool()
def generate_frida_script(target: Annotated[str, Field(description='目标函数名或地址')], script_type: Annotated[str, Field(description="脚本类型: 'hook', 'memory_dump', 'string_hook'")], options: Annotated[dict, Field(description='可选配置参数')]=None) -> str:
    """
    生成frida动态分析脚本。
    参数：
        target: 目标函数名或地址
        script_type: 脚本类型
        options: 可选配置
    返回：
        生成的JavaScript脚本代码
    """
    return make_jsonrpc_request('generate_frida_script', target, script_type, options)

@mcp.tool()
def save_generated_script(script_content: Annotated[str, Field(description='脚本内容')], script_type: Annotated[str, Field(description="脚本类型: 'angr' 或 'frida'")], file_name: Annotated[str, Field(description='保存的文件名（可选）')]=None) -> str:
    """
    保存生成的脚本到文件。
    参数：
        script_content: 脚本内容
        script_type: 脚本类型
        file_name: 保存的文件名（可选）
    返回：
        保存的文件路径
    """
    return make_jsonrpc_request('save_generated_script', script_content, script_type, file_name)

@mcp.tool()
def run_external_script(script_path: Annotated[str, Field(description='脚本文件路径')], script_type: Annotated[str, Field(description="脚本类型: 'angr' 或 'frida'")], target_binary: Annotated[str, Field(description='目标二进制文件路径')]=None, target_pid: Annotated[int, Field(description='目标进程ID（可选，用于frida）')]=None) -> str:
    """
    运行外部脚本（angr或frida）。
    参数：
        script_path: 脚本文件路径
        script_type: 脚本类型
        target_binary: 目标二进制文件路径（可选）
        target_pid: 目标进程ID（可选，用于frida附加到运行中的进程）
    返回：
        脚本执行的输出结果
    """
    return make_jsonrpc_request('run_external_script', script_path, script_type, target_binary, target_pid)

@mcp.tool()
def get_function_call_graph(start_address: Annotated[str, Field(description='起始函数地址')], depth: Annotated[int, Field(description='递归深度')]=3, mermaid: Annotated[bool, Field(description='是否返回 mermaid 格式')]=False) -> dict:
    """
    获取函数调用图。
    参数：
        start_address: 起始函数地址（字符串）
        depth: 递归深度，默认3
        mermaid: 是否返回 mermaid 格式（默认False，返回邻接表）
    返回：
        {"graph": 邻接表或mermaid字符串, "nodes": 节点列表, "edges": 边列表}
    """
    return make_jsonrpc_request('get_function_call_graph', start_address, depth, mermaid)

@mcp.tool()
def get_analysis_report() -> dict:
    """
    自动生成结构化分析报告。
    """
    return make_jsonrpc_request('get_analysis_report')

@mcp.tool()
def get_incremental_changes() -> list:
    """
    返回自上次分析以来的增量变更。
    """
    return make_jsonrpc_request('get_incremental_changes')

@mcp.tool()
def get_dynamic_string_map() -> dict:
    """
    动态字符串解密映射（静态+动态分析结果）。
    """
    return make_jsonrpc_request('get_dynamic_string_map')

@mcp.tool()
def generate_analysis_report_md() -> str:
    """
    一键生成结构化 markdown 报告，帮助用户快速理解程序核心逻辑。
    """
    return make_jsonrpc_request('generate_analysis_report_md')

