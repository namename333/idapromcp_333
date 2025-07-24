import os
import sys

if sys.version_info < (3, 11):
    raise RuntimeError("MCP 插件需要 Python 3.11 及以上版本")

import json
import struct
import threading
import http.server
from urllib.parse import urlparse
from typing import Any, Callable, get_type_hints, TypedDict, Optional, Annotated, TypeVar, Generic, NotRequired
import re
import time
import tempfile
import subprocess

# JSON-RPC 错误类
class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data

# RPC 注册表，管理所有可调用方法
class RPCRegistry:
    def __init__(self):
        self.methods: dict[str, Callable] = {}
        self.unsafe: set[str] = set()

    def register(self, func: Callable) -> Callable:
        self.methods[func.__name__] = func
        return func

    def mark_unsafe(self, func: Callable) -> Callable:
        self.unsafe.add(func.__name__)
        return func

    def dispatch(self, method: str, params: Any) -> Any:
        if method not in self.methods:
            raise JSONRPCError(-32601, f"方法 '{method}' 未找到")

        func = self.methods[method]
        hints = get_type_hints(func)

        # 移除返回值注解
        hints.pop("return", None)

        if isinstance(params, list):
            if len(params) != len(hints):
                raise JSONRPCError(-32602, f"参数数量错误: 期望 {len(hints)} 个，实际 {len(params)} 个")

            # 参数类型校验与转换
            converted_params = []
            for value, (param_name, expected_type) in zip(params, hints.items()):
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params.append(value)
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"参数 '{param_name}' 类型错误: 期望 {expected_type.__name__}")

            return func(*converted_params)
        elif isinstance(params, dict):
            if set(params.keys()) != set(hints.keys()):
                raise JSONRPCError(-32602, f"参数名错误: 期望 {list(hints.keys())}")

            # 参数类型校验与转换
            converted_params = {}
            for param_name, expected_type in hints.items():
                value = params.get(param_name)
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params[param_name] = value
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"参数 '{param_name}' 类型错误: 期望 {expected_type.__name__}")

            return func(**converted_params)
        else:
            raise JSONRPCError(-32600, "请求参数必须为数组或对象")

rpc_registry = RPCRegistry()

# 注册 JSON-RPC 方法的装饰器
def jsonrpc(func: Callable) -> Callable:
    global rpc_registry
    return rpc_registry.register(func)

# 标记为不安全方法的装饰器
def unsafe(func: Callable) -> Callable:
    return rpc_registry.mark_unsafe(func)

# JSON-RPC 请求处理器
class JSONRPCRequestHandler(http.server.BaseHTTPRequestHandler):
    def send_jsonrpc_error(self, code: int, message: str, id: Any = None):
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            }
        }
        if id is not None:
            response["id"] = id
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def do_POST(self):
        global rpc_registry

        parsed_path = urlparse(self.path)
        if parsed_path.path != "/mcp":
            self.send_jsonrpc_error(-32098, "无效的接口路径", None)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_jsonrpc_error(-32700, "请求体缺失", None)
            return

        request_body = self.rfile.read(content_length)
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_jsonrpc_error(-32700, "JSON 解析错误", None)
            return

        # 构造响应内容
        response = {
            "jsonrpc": "2.0"
        }
        if request.get("id") is not None:
            response["id"] = request.get("id")

        try:
            # 基本 JSON-RPC 校验
            if not isinstance(request, dict):
                raise JSONRPCError(-32600, "请求格式错误")
            if request.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "JSON-RPC 版本错误")
            if "method" not in request:
                raise JSONRPCError(-32600, "未指定方法名")

            # 分发方法调用
            result = rpc_registry.dispatch(request["method"], request.get("params", []))
            response["result"] = result

        except JSONRPCError as e:
            response["error"] = {
                "code": e.code,
                "message": e.message
            }
            if e.data is not None:
                response["error"]["data"] = e.data
        except IDAError as e:
            response["error"] = {
                "code": -32000,
                "message": e.message,
            }
        except Exception as e:
            traceback.print_exc()
            response["error"] = {
                "code": -32603,
                "message": "内部错误（请反馈 bug）",
                "data": traceback.format_exc(),
            }

        try:
            response_body = json.dumps(response).encode("utf-8")
        except Exception as e:
            traceback.print_exc()
            response_body = json.dumps({
                "error": {
                    "code": -32603,
                    "message": "内部错误（请反馈 bug）",
                    "data": traceback.format_exc(),
                }
            }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        # 屏蔽 HTTP 日志输出
        pass

# MCP HTTP 服务器
class MCPHTTPServer(http.server.HTTPServer):
    allow_reuse_address = False

# 服务器主类
class Server:
    HOST = "localhost"
    PORT = 13337

    def __init__(self):
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        if self.running:
            print("[MCP] 服务器已在运行")
            return

        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.running = True
        self.server_thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
            self.server = None
        print("[MCP] 服务器已停止")

    def _run_server(self):
        try:
            # 在线程中启动 HTTP 服务器
            self.server = MCPHTTPServer((Server.HOST, Server.PORT), JSONRPCRequestHandler)
            print(f"[MCP] 服务器已启动: http://{Server.HOST}:{Server.PORT}")
            self.server.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:  # 端口被占用（Linux/Windows）
                print("[MCP] 错误: 端口 13337 已被占用")
            else:
                print(f"[MCP] 服务器错误: {e}")
            self.running = False
        except Exception as e:
            print(f"[MCP] 服务器错误: {e}")
        finally:
            self.running = False

# 一个帮助编写线程安全IDA代码的模块。
# Based on:
# https://web.archive.org/web/20160305190440/http://www.williballenthin.com/blog/2015/09/04/idapython-synchronization-decorator/
import logging
import queue
import traceback
import functools

import ida_hexrays
import ida_kernwin
import ida_funcs
import ida_gdl
import ida_lines
import ida_idaapi
import idc
import idaapi
import idautils
import ida_nalt
import ida_bytes
import ida_typeinf
import ida_xref
import ida_entry
import idautils
import ida_idd
import ida_dbg
import ida_name
import ida_ida
import ida_frame

class IDAError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]

class IDASyncError(Exception):
    pass

class DecompilerLicenseError(IDAError):
    pass

# 重要提示：始终确保函数 f 的返回值是从 IDA 获取的数据的副本，而不是原始数据。
#
# 示例：
# --------
#
# 正确做法：
#
#   @idaread
#   def ts_Functions():
#       return list(idautils.Functions())
#
# 错误做法：
#
#   @idaread
#   def ts_Functions():
#       return idautils.Functions()
#

logger = logging.getLogger(__name__)

# 安全模式枚举，数值越高表示越安全：
class IDASafety:
    ida_kernwin.MFF_READ
    SAFE_NONE = ida_kernwin.MFF_FAST
    SAFE_READ = ida_kernwin.MFF_READ
    SAFE_WRITE = ida_kernwin.MFF_WRITE

call_stack = queue.LifoQueue()

def sync_wrapper(ff, safety_mode: IDASafety):
    """
    调用一个函数 ff 并指定一个特定的 IDA 安全模式。
    """
    #logger.debug('sync_wrapper: {}, {}'.format(ff.__name__, safety_mode))  # 调试日志：同步包装器信息

    if safety_mode not in [IDASafety.SAFE_READ, IDASafety.SAFE_WRITE]:
        error_str = 'Invalid safety mode {} over function {}'\
                .format(safety_mode, ff.__name__)
        logger.error(error_str)
        raise IDASyncError(error_str)

    # 未设置安全级别：
    res_container = queue.Queue()

    def runned():
        #logger.debug('Inside runned')  # 调试日志：进入运行状态

        # 确保我们不在sync_wrapper内部：
        if not call_stack.empty():
            last_func_name = call_stack.get()
            error_str = ('Call stack is not empty while calling the '
                'function {} from {}').format(ff.__name__, last_func_name)
            #logger.error(error_str)  # 错误日志：输出错误信息
            raise IDASyncError(error_str)

        call_stack.put((ff.__name__))
        try:
            res_container.put(ff())
        except Exception as x:
            res_container.put(x)
        finally:
            call_stack.get()
            #logger.debug('Finished runned')

    ret_val = idaapi.execute_sync(runned, safety_mode)
    res = res_container.get()
    if isinstance(res, Exception):
        raise res
    return res

def idawrite(f):
    """
    标记一个函数为修改 IDB 的装饰器。
    在主 IDA 循环中安排一个请求，以避免 IDB 损坏。
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_WRITE)
    return wrapper

def idaread(f):
    """
    标记一个函数为从 IDB 读取的装饰器。
    在主 IDA 循环中安排一个请求，以避免
     不一致的结果。
    MFF_READ 常量通过：http://www.openrce.org/forums/posts/1827
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_READ)
    return wrapper

def is_window_active():
    """返回 IDA 当前是否处于活动状态"""
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        return False

    app = QApplication.instance()
    if app is None:
        return False

    for widget in app.topLevelWidgets():
        if widget.isActiveWindow():
            return True
    return False

class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str

def get_image_size() -> int:
    try:
        # https://www.hex-rays.com/products/ida/support/sdkdoc/structidainfo.html
        info = idaapi.get_inf_structure()
        omin_ea = info.omin_ea
        omax_ea = info.omax_ea
    except AttributeError:
        import ida_ida
        omin_ea = ida_ida.inf_get_omin_ea()
        omax_ea = ida_ida.inf_get_omax_ea()
    # 一个不准确的图像大小（如果重定位在最后）
    image_size = omax_ea - omin_ea
    # 尝试从 PE 头中提取它
    header = idautils.peutils_t().header()
    if header and header[:4] == b"PE\0\0":
        image_size = struct.unpack("<I", header[0x50:0x54])[0]
    return image_size

@jsonrpc
@idaread
def get_metadata() -> Metadata:
    """获取当前 IDB 的元数据"""
    # Fat Mach-O 二进制文件可以返回 None 哈希：
    # https://github.com/mrexodia/ida-pro-mcp/issues/26
    def hash(f):
        try:
            return f().hex()
        except:
            return None

    return Metadata(path=idaapi.get_input_file_path(),
                    module=idaapi.get_root_filename(),
                    base=hex(idaapi.get_imagebase()),
                    size=hex(get_image_size()),
                    md5=hash(ida_nalt.retrieve_input_file_md5),
                    sha256=hash(ida_nalt.retrieve_input_file_sha256),
                    crc32=hex(ida_nalt.retrieve_input_file_crc32()),
                    filesize=hex(ida_nalt.retrieve_input_file_size()))

def get_prototype(fn: ida_funcs.func_t) -> Optional[str]:
    try:
        prototype: ida_typeinf.tinfo_t = fn.get_prototype()
        if prototype is not None:
            return str(prototype)
        else:
            return None
    except AttributeError:
        try:
            return idc.get_type(fn.start_ea)
        except:
            tif = ida_typeinf.tinfo_t()
            if ida_nalt.get_tinfo(tif, fn.start_ea):
                return str(tif)
            return None
    except Exception as e:
        print(f"Error getting function prototype: {e}")
        return None

class Function(TypedDict):
    address: str
    name: str
    size: str

def parse_address(address: str) -> int:
    try:
        return int(address, 0)
    except ValueError:
        for ch in address:
            if ch not in "0123456789abcdefABCDEF":
                raise IDAError(f"Failed to parse address: {address}")
        raise IDAError(f"Failed to parse address (missing 0x prefix): {address}")

def get_function(address: int, *, raise_error=True) -> Function:
    fn = idaapi.get_func(address)
    if fn is None:
        if raise_error:
            raise IDAError(f"No function found at address {hex(address)}")
        return None

    try:
        name = fn.get_name()
    except AttributeError:
        name = ida_funcs.get_func_name(fn.start_ea)

    return Function(address=hex(address), name=name, size=hex(fn.end_ea - fn.start_ea))

DEMANGLED_TO_EA = {}

def create_demangled_to_ea_map():
    for ea in idautils.Functions():
        # 获取函数名并进行解混淆
        # MNG_NODEFINIT 标志仅保留主名称，抑制其他信息
        # 默认解混淆会添加函数签名
        # 以及装饰器（如有）
        demangled = idaapi.demangle_name(
            idc.get_name(ea, 0), idaapi.MNG_NODEFINIT)
        if demangled:
            DEMANGLED_TO_EA[demangled] = ea


def get_type_by_name(type_name: str) -> ida_typeinf.tinfo_t:
    # 8-bit integers
    if type_name in ('int8', '__int8', 'int8_t', 'char', 'signed char'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT8)
    elif type_name in ('uint8', '__uint8', 'uint8_t', 'unsigned char', 'byte', 'BYTE'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT8)

    # 16-bit integers
    elif type_name in ('int16', '__int16', 'int16_t', 'short', 'short int', 'signed short', 'signed short int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT16)
    elif type_name in ('uint16', '__uint16', 'uint16_t', 'unsigned short', 'unsigned short int', 'word', 'WORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT16)

    # 32-bit integers
    elif type_name in ('int32', '__int32', 'int32_t', 'int', 'signed int', 'long', 'long int', 'signed long', 'signed long int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT32)
    elif type_name in ('uint32', '__uint32', 'uint32_t', 'unsigned int', 'unsigned long', 'unsigned long int', 'dword', 'DWORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT32)

    # 64-bit integers
    elif type_name in ('int64', '__int64', 'int64_t', 'long long', 'long long int', 'signed long long', 'signed long long int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT64)
    elif type_name in ('uint64', '__uint64', 'uint64_t', 'unsigned int64', 'unsigned long long', 'unsigned long long int', 'qword', 'QWORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT64)

    # 128-bit integers
    elif type_name in ('int128', '__int128', 'int128_t', '__int128_t'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT128)
    elif type_name in ('uint128', '__uint128', 'uint128_t', '__uint128_t', 'unsigned int128'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT128)

    # 浮点类型
    elif type_name in ('float', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_FLOAT)
    elif type_name in ('double', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_DOUBLE)
    elif type_name in ('long double', 'ldouble'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_LDOUBLE)

    # 布尔类型
    elif type_name in ('bool', '_Bool', 'boolean'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_BOOL)

    # 空类型
    elif type_name in ('void', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_VOID)

    # 如果不是标准类型，尝试获取命名类型
    tif = ida_typeinf.tinfo_t()
    if tif.get_named_type(None, type_name, ida_typeinf.BTF_STRUCT):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_TYPEDEF):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_ENUM):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_UNION):
        return tif

    if tif := ida_typeinf.tinfo_t(type_name):
        return tif

    raise IDAError(f"Unable to retrieve {type_name} type info object")

@jsonrpc
@idaread
def get_function_by_name(
    name: Annotated[str, "要获取的函数名称"]
) -> Function:
    """根据函数名称获取函数"""
    function_address = idaapi.get_name_ea(idaapi.BADADDR, name)
    if function_address == idaapi.BADADDR:
        # 如果映射尚未创建，则创建它
        if len(DEMANGLED_TO_EA) == 0:
            create_demangled_to_ea_map()
        # 尝试在映射中查找函数，否则抛出错误
        if name in DEMANGLED_TO_EA:
            function_address = DEMANGLED_TO_EA[name]
        else:
            raise IDAError(f"No function found with name {name}")
    return get_function(function_address)

@jsonrpc
@idaread
def get_function_by_address(
    address: Annotated[str, "要获取的函数地址"],
) -> Function:
    """根据函数地址获取函数"""
    return get_function(parse_address(address))

@jsonrpc
@idaread
def get_current_address() -> str:
    """获取用户当前选中的地址"""
    return hex(idaapi.get_screen_ea())

@jsonrpc
@idaread
def get_current_function() -> Optional[Function]:
    """获取用户当前选中的函数"""
    return get_function(idaapi.get_screen_ea())

class ConvertedNumber(TypedDict):
    decimal: str
    hexadecimal: str
    bytes: str
    ascii: Optional[str]
    binary: str

@jsonrpc
def convert_number(
    text: Annotated[str, "要转换的数字的文本表示"],
    size: Annotated[Optional[int], "变量的大小（字节）"],
) -> ConvertedNumber:
    """将数字（十进制、十六进制）转换为不同表示"""
    try:
        value = int(text, 0)
    except ValueError:
        raise IDAError(f"Invalid number: {text}")

    # 估计数字的大小
    if not size:
        size = 0
        n = abs(value)
        while n:
            size += 1
            n >>= 1
        size += 7
        size //= 8

    # 将数字转换为字节
    try:
        bytes = value.to_bytes(size, "little", signed=True)
    except OverflowError:
        raise IDAError(f"Number {text} is too big for {size} bytes")

    # 将字节转换为 ASCII
    ascii = ""
    for byte in bytes.rstrip(b"\x00"):
        if byte >= 32 and byte <= 126:
            ascii += chr(byte)
        else:
            ascii = None
            break

    return ConvertedNumber(
        decimal=str(value),
        hexadecimal=hex(value),
        bytes=bytes.hex(" "),
        ascii=ascii,
        binary=bin(value),
    )

T = TypeVar("T")

class Page(TypedDict, Generic[T]):
    data: list[T]
    next_offset: Optional[int]

def paginate(data: list[T], offset: int, count: int) -> Page[T]:
    if count == 0:
        count = len(data)
    next_offset = offset + count
    if next_offset >= len(data):
        next_offset = None
    return {
        "data": data[offset:offset + count],
        "next_offset": next_offset,
    }

def pattern_filter(data: list[T], pattern: str, key: str) -> list[T]:
    if not pattern:
        return data

    # TODO: implement /regex/ matching

    def matches(item: T) -> bool:
        return pattern.lower() in item[key].lower()
    return list(filter(matches, data))

@jsonrpc
@idaread
def list_functions(
    offset: Annotated[int, "从 (0) 开始列出偏移量"],
    count: Annotated[int, "要列出的函数数量 (100 是默认值，0 表示剩余)"],
) -> Page[Function]:
    """列出数据库中的所有函数（分页）"""
    functions = [get_function(address) for address in idautils.Functions()]
    return paginate(functions, offset, count)

class Global(TypedDict):
    address: str
    name: str

@jsonrpc
@idaread
def list_globals_filter(
    offset: Annotated[int, "从 (0) 开始列出偏移量"],
    count: Annotated[int, "要列出的全局变量数量 (100 是默认值，0 表示剩余)"],
    filter: Annotated[str, "要应用的过滤器 (必需参数，空字符串表示无过滤). 大小写不敏感包含或 /regex/ 语法"],
) -> Page[Global]:
    """列出数据库中的匹配全局变量（分页，过滤）"""
    globals = []
    for addr, name in idautils.Names():
        # 跳过函数
        if not idaapi.get_func(addr):
            globals += [Global(address=hex(addr), name=name)]

    globals = pattern_filter(globals, filter, "name")
    return paginate(globals, offset, count)

@jsonrpc
def list_globals(
    offset: Annotated[int, "从 (0) 开始列出偏移量"],
    count: Annotated[int, "要列出的全局变量数量 (100 是默认值，0 表示剩余)"],
) -> Page[Global]:
    """列出数据库中的所有全局变量（分页）"""
    return list_globals_filter(offset, count, "")

class Import(TypedDict):
    address: str
    imported_name: str
    module: str

@jsonrpc
@idaread
def list_imports(
        offset: Annotated[int, "从 (0) 开始列出偏移量"],
        count: Annotated[int, "要列出的导入符号数量 (100 是默认值，0 表示剩余)"],
) -> Page[Import]:
    """ 列出所有导入符号及其名称和模块（分页） """
    nimps = ida_nalt.get_import_module_qty()

    rv = []
    for i in range(nimps):
        module_name = ida_nalt.get_import_module_name(i)
        if not module_name:
            module_name = "<unnamed>"

        def imp_cb(ea, symbol_name, ordinal, acc):
            if not symbol_name:
                symbol_name = f"#{ordinal}"

            acc += [Import(address=hex(ea), imported_name=symbol_name, module=module_name)]

            return True

        imp_cb_w_context = lambda ea, symbol_name, ordinal: imp_cb(ea, symbol_name, ordinal, rv)
        ida_nalt.enum_import_names(i, imp_cb_w_context)

    return paginate(rv, offset, count)

class String(TypedDict):
    address: str
    length: int
    string: str

@jsonrpc
@idaread
def list_strings_filter(
    offset: Annotated[int, "从 (0) 开始列出偏移量"],
    count: Annotated[int, "要列出的字符串数量 (100 是默认值，0 表示剩余)"],
    filter: Annotated[str, "要应用的过滤器 (必需参数，空字符串表示无过滤). 大小写不敏感包含或 /regex/ 语法"],
) -> Page[String]:
    """列出数据库中的匹配字符串（分页，过滤）"""
    strings = []
    for item in idautils.Strings():
        try:
            string = str(item)
            if string:
                strings += [
                    String(address=hex(item.ea), length=item.length, string=string),
                ]
        except:
            continue
    strings = pattern_filter(strings, filter, "string")
    return paginate(strings, offset, count)

@jsonrpc
def list_strings(
    offset: Annotated[int, "从 (0) 开始列出偏移量"],
    count: Annotated[int, "要列出的字符串数量 (100 是默认值，0 表示剩余)"],
) -> Page[String]:
    """列出数据库中的所有字符串（分页）"""
    return list_strings_filter(offset, count, "")

@jsonrpc
@idaread
def list_local_types():
    """列出数据库中的所有本地类型"""
    error = ida_hexrays.hexrays_failure_t()
    locals = []
    idati = ida_typeinf.get_idati()
    type_count = ida_typeinf.get_ordinal_limit(idati)
    for ordinal in range(1, type_count):
        try:
            tif = ida_typeinf.tinfo_t()
            if tif.get_numbered_type(idati, ordinal):
                type_name = tif.get_type_name()
                if not type_name:
                    type_name = f"<Anonymous Type #{ordinal}>"
                locals.append(f"\nType #{ordinal}: {type_name}")
                if tif.is_udt():
                    c_decl_flags = (ida_typeinf.PRTYPE_MULTI | ida_typeinf.PRTYPE_TYPE | ida_typeinf.PRTYPE_SEMI | ida_typeinf.PRTYPE_DEF | ida_typeinf.PRTYPE_METHODS | ida_typeinf.PRTYPE_OFFSETS)
                    c_decl_output = tif._print(None, c_decl_flags)
                    if c_decl_output:
                        locals.append(f"  C declaration:\n{c_decl_output}")
                else:
                    simple_decl = tif._print(None, ida_typeinf.PRTYPE_1LINE | ida_typeinf.PRTYPE_TYPE | ida_typeinf.PRTYPE_SEMI)
                    if simple_decl:
                        locals.append(f"  Simple declaration:\n{simple_decl}")  
            else:
                message = f"\nType #{ordinal}: Failed to retrieve information."
                if error.str:
                    message += f": {error.str}"
                if error.errea != idaapi.BADADDR:
                    message += f"from (address: {hex(error.errea)})"
                raise IDAError(message)
        except:
            continue
    return locals

def decompile_checked(address: int) -> ida_hexrays.cfunc_t:
    if not ida_hexrays.init_hexrays_plugin():
        raise IDAError("Hex-Rays 反编译器不可用")
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(address, error, ida_hexrays.DECOMP_WARNINGS)
    if not cfunc:
        if error.code == ida_hexrays.MERR_LICENSE:
            raise DecompilerLicenseError("反编译器许可证不可用。请使用 `disassemble_function` 获取汇编代码。")

        message = f"Decompilation failed at {hex(address)}"
        if error.str:
            message += f": {error.str}"
        if error.errea != idaapi.BADADDR:
            message += f" (address: {hex(error.errea)})"
        raise IDAError(message)
    return cfunc

@jsonrpc
@idaread
def decompile_function(
    address: Annotated[str, "要反编译的函数地址"],
) -> str:
    """反编译给定地址的函数"""
    address = parse_address(address)
    cfunc = decompile_checked(address)
    if is_window_active():
        ida_hexrays.open_pseudocode(address, ida_hexrays.OPF_REUSE)
    sv = cfunc.get_pseudocode()
    pseudocode = ""
    for i, sl in enumerate(sv):
        sl: ida_kernwin.simpleline_t
        item = ida_hexrays.ctree_item_t()
        addr = None if i > 0 else cfunc.entry_ea
        if cfunc.get_line_item(sl.line, 0, False, None, item, None):
            ds = item.dstr().split(": ")
            if len(ds) == 2:
                try:
                    addr = int(ds[0], 16)
                except ValueError:
                    pass
        line = ida_lines.tag_remove(sl.line)
        if len(pseudocode) > 0:
            pseudocode += "\n"
        if not addr:
            pseudocode += f"/* line: {i} */ {line}"
        else:
            pseudocode += f"/* line: {i}, address: {hex(addr)} */ {line}"

    return pseudocode

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

@jsonrpc
@idaread
def disassemble_function(
    start_address: Annotated[str, "要反汇编的函数地址"],
) -> DisassemblyFunction:
    """获取函数汇编代码"""
    start = parse_address(start_address)
    func: ida_funcs.func_t = idaapi.get_func(start)
    if not func:
        raise IDAError(f"No function found containing address {start_address}")
    if is_window_active():
        ida_kernwin.jumpto(start)

    lines = []
    for address in ida_funcs.func_item_iterator_t(func):
        seg = idaapi.getseg(address)
        segment = idaapi.get_segm_name(seg) if seg else None

        label = idc.get_name(address, 0)
        if label and label == func.name and address == func.start_ea:
            label = None
        if label == "":
            label = None

        comments = []
        if comment := idaapi.get_cmt(address, False):
            comments += [comment]
        if comment := idaapi.get_cmt(address, True):
            comments += [comment]

        raw_instruction = idaapi.generate_disasm_line(address, 0)
        tls = ida_kernwin.tagged_line_sections_t()
        ida_kernwin.parse_tagged_line_sections(tls, raw_instruction)
        insn_section = tls.first(ida_lines.COLOR_INSN)

        operands = []
        for op_tag in range(ida_lines.COLOR_OPND1, ida_lines.COLOR_OPND8 + 1):
            op_n = tls.first(op_tag)
            if not op_n:
                break

            op: str = op_n.substr(raw_instruction)
            op_str = ida_lines.tag_remove(op)

            # 做很多工作来添加地址注释以获取符号
            for idx in range(len(op) - 2):
                if op[idx] != idaapi.COLOR_ON:
                    continue

                idx += 1
                if ord(op[idx]) != idaapi.COLOR_ADDR:
                    continue

                idx += 1
                addr_string = op[idx:idx + idaapi.COLOR_ADDR_SIZE]
                idx += idaapi.COLOR_ADDR_SIZE

                addr = int(addr_string, 16)

                # 找到下一个颜色并切片直到那里
                symbol = op[idx:op.find(idaapi.COLOR_OFF, idx)]

                if symbol == '':
                    # 我们无法确定符号，所以使用整个 op_str
                    symbol = op_str

                comments += [f"{symbol}={addr:#x}"]

                # 如果其类型可用，则打印其值
                try:
                    value = get_global_variable_value_internal(addr)
                except:
                    continue

                comments += [f"*{symbol}={value}"]

            operands += [op_str]

        mnem = ida_lines.tag_remove(insn_section.substr(raw_instruction))
        instruction = f"{mnem} {', '.join(operands)}"

        line = DisassemblyLine(
            address=f"{address:#x}",
            instruction=instruction,
        )

        if len(comments) > 0:
            line.update(comments=comments)

        if segment:
            line.update(segment=segment)

        if label:
            line.update(label=label)

        lines += [line]

    prototype = func.get_prototype()
    arguments: list[Argument] = [Argument(name=arg.name, type=f"{arg.type}") for arg in prototype.iter_func()] if prototype else None

    disassembly_function = DisassemblyFunction(
        name=func.name,
        start_ea=f"{func.start_ea:#x}",
        stack_frame=get_stack_frame_variables_internal(func.start_ea),
        lines=lines
    )

    if prototype:
        disassembly_function.update(return_type=f"{prototype.get_rettype()}")

    if arguments:
        disassembly_function.update(arguments=arguments)

    return disassembly_function

class Xref(TypedDict):
    address: str
    type: str
    function: Optional[Function]

@jsonrpc
@idaread
def get_xrefs_to(
    address: Annotated[str, "要获取交叉引用的地址"],
) -> list[Xref]:
    """获取给定地址的所有交叉引用"""
    xrefs = []
    xref: ida_xref.xrefblk_t
    for xref in idautils.XrefsTo(parse_address(address)):
        xrefs += [
            Xref(address=hex(xref.frm),
                 type="code" if xref.iscode else "data",
                 function=get_function(xref.frm, raise_error=False))
        ]
    return xrefs

@jsonrpc
@idaread
def get_xrefs_to_field(
    struct_name: Annotated[str, "结构体名称 (类型)"],
    field_name: Annotated[str, "要获取交叉引用的字段名称 (成员)"],
) -> list[Xref]:
    """获取命名结构体字段 (成员) 的所有交叉引用"""

    # 获取类型库
    til = ida_typeinf.get_idati()
    if not til:
        raise IDAError("Failed to retrieve type library.")

    # 获取结构体类型信息
    tif = ida_typeinf.tinfo_t()
    if not tif.get_named_type(til, struct_name, ida_typeinf.BTF_STRUCT, True, False):
        print(f"Structure '{struct_name}' not found.")
        return []

    # 获取字段索引
    idx = ida_typeinf.get_udm_by_fullname(None, struct_name + '.' + field_name)
    if idx == -1:
        print(f"Field '{field_name}' not found in structure '{struct_name}'.")
        return []

    # 获取类型标识符
    tid = tif.get_udm_tid(idx)
    if tid == ida_idaapi.BADADDR:
        raise IDAError(f"Unable to get tid for structure '{struct_name}' and field '{field_name}'.")

    # 获取 tid 的交叉引用
    xrefs = []
    xref: ida_xref.xrefblk_t
    for xref in idautils.XrefsTo(tid):

        xrefs += [
            Xref(address=hex(xref.frm),
                 type="code" if xref.iscode else "data",
                 function=get_function(xref.frm, raise_error=False))
        ]
    return xrefs

@jsonrpc
@idaread
def get_entry_points() -> list[Function]:
    """获取数据库中的所有入口点"""
    result = []
    for i in range(ida_entry.get_entry_qty()):
        ordinal = ida_entry.get_entry_ordinal(i)
        address = ida_entry.get_entry(ordinal)
        func = get_function(address, raise_error=False)
        if func is not None:
            result.append(func)
    return result

@jsonrpc
@idawrite
def set_comment(
    address: Annotated[str, "要设置注释的函数地址"],
    comment: Annotated[str, "注释文本"],
):
    """设置给定函数反汇编和伪代码中的注释"""
    address = parse_address(address)

    if not idaapi.set_cmt(address, comment, False):
        raise IDAError(f"Failed to set disassembly comment at {hex(address)}")

    if not ida_hexrays.init_hexrays_plugin():
        return

    # 参考：https://cyber.wtf/2019/03/22/using-ida-python-to-analyze-trickbot/
    # 检查地址是否对应于一行
    try:
        cfunc = decompile_checked(address)
    except DecompilerLicenseError:
        # 由于反编译器许可证错误，我们未能反编译函数
        return

    # 特殊情况：函数入口注释
    if address == cfunc.entry_ea:
        idc.set_func_cmt(address, comment, True)
        cfunc.refresh_func_ctext()
        return

    eamap = cfunc.get_eamap()
    if address not in eamap:
        print(f"Failed to set decompiler comment at {hex(address)}")
        return
    nearest_ea = eamap[address][0].ea

    # 移除孤立注释
    if cfunc.has_orphan_cmts():
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()

    # 尝试所有可能的项目类型设置注释
    tl = idaapi.treeloc_t()
    tl.ea = nearest_ea
    for itp in range(idaapi.ITP_SEMI, idaapi.ITP_COLON):
        tl.itp = itp
        cfunc.set_user_cmt(tl, comment)
        cfunc.save_user_cmts()
        cfunc.refresh_func_ctext()
        if not cfunc.has_orphan_cmts():
            return
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()
    print(f"Failed to set decompiler comment at {hex(address)}")

def refresh_decompiler_widget():
    widget = ida_kernwin.get_current_widget()
    if widget is not None:
        vu = ida_hexrays.get_widget_vdui(widget)
        if vu is not None:
            vu.refresh_ctext()

def refresh_decompiler_ctext(function_address: int):
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(function_address, error, ida_hexrays.DECOMP_WARNINGS)
    if cfunc:
        cfunc.refresh_func_ctext()

@jsonrpc
@idawrite
def rename_local_variable(
    function_address: Annotated[str, "包含变量的函数地址"],
    old_name: Annotated[str, "变量的当前名称"],
    new_name: Annotated[str, "变量的新名称 (空表示默认名称)"],
):
    """重命名函数中的本地变量"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, old_name, new_name):
        raise IDAError(f"Failed to rename local variable {old_name} in function {hex(func.start_ea)}")
    refresh_decompiler_ctext(func.start_ea)

@jsonrpc
@idawrite
def rename_global_variable(
    old_name: Annotated[str, "全局变量的当前名称"],
    new_name: Annotated[str, "全局变量的新名称 (空表示默认名称)"],
):
    """重命名全局变量"""
    ea = idaapi.get_name_ea(idaapi.BADADDR, old_name)
    if not idaapi.set_name(ea, new_name):
        raise IDAError(f"Failed to rename global variable {old_name} to {new_name}")
    refresh_decompiler_ctext(ea)

@jsonrpc
@idawrite
def set_global_variable_type(
    variable_name: Annotated[str, "全局变量的名称"],
    new_type: Annotated[str, "变量的新类型"],
):
    """设置全局变量的类型"""
    ea = idaapi.get_name_ea(idaapi.BADADDR, variable_name)
    tif = get_type_by_name(new_type)
    if not tif:
        raise IDAError(f"Parsed declaration is not a variable type")
    if not ida_typeinf.apply_tinfo(ea, tif, ida_typeinf.PT_SIL):
        raise IDAError(f"Failed to apply type")

@jsonrpc
@idaread
def get_global_variable_value_by_name(variable_name: Annotated[str, "全局变量的名称"]) -> str:
    """
    读取全局变量的值（如果编译时已知）

    优先使用此函数，而不是 `data_read_*` 函数。
    """
    ea = idaapi.get_name_ea(idaapi.BADADDR, variable_name)
    if ea == idaapi.BADADDR:
        raise IDAError(f"Global variable {variable_name} not found")

    return get_global_variable_value_internal(ea)

@jsonrpc
@idaread
def get_global_variable_value_at_address(ea: Annotated[str, "全局变量的地址"]) -> str:
    """
    通过地址读取全局变量的值（如果编译时已知）

    优先使用此函数，而不是 `data_read_*` 函数。
    """
    ea = parse_address(ea)
    return get_global_variable_value_internal(ea)

def get_global_variable_value_internal(ea: int) -> str:
     # 获取变量的类型信息
     tif = ida_typeinf.tinfo_t()
     if not ida_nalt.get_tinfo(tif, ea):
         # 没有类型信息，也许我们可以通过名称推断其大小
         if not ida_bytes.has_any_name(ea):
             raise IDAError(f"Failed to get type information for variable at {ea:#x}")

         size = ida_bytes.get_item_size(ea)
         if size == 0:
             raise IDAError(f"Failed to get type information for variable at {ea:#x}")
     else:
         # 确定变量的大小
         size = tif.get_size()

     # 根据大小读取值
     if size == 0 and tif.is_array() and tif.get_array_element().is_decl_char():
         return_string = idaapi.get_strlit_contents(ea, -1, 0).decode("utf-8").strip()
         return f"\"{return_string}\""
     elif size == 1:
         return hex(ida_bytes.get_byte(ea))
     elif size == 2:
         return hex(ida_bytes.get_word(ea))
     elif size == 4:
         return hex(ida_bytes.get_dword(ea))
     elif size == 8:
         return hex(ida_bytes.get_qword(ea))
     else:
         # 对于其他大小，返回原始字节
         return ' '.join(hex(x) for x in ida_bytes.get_bytes(ea, size))


@jsonrpc
@idawrite
def rename_function(
    function_address: Annotated[str, "要重命名的函数地址"],
    new_name: Annotated[str, "函数的新名称 (空表示默认名称)"],
):
    """重命名函数"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not idaapi.set_name(func.start_ea, new_name):
        raise IDAError(f"Failed to rename function {hex(func.start_ea)} to {new_name}")
    refresh_decompiler_ctext(func.start_ea)
    # 自动记录变更
    record_incremental_change("rename_function", {"address": function_address, "new_name": new_name})

@jsonrpc
@idawrite
def set_function_prototype(
    function_address: Annotated[str, "函数地址"],
    prototype: Annotated[str, "新的函数原型"],
):
    """设置函数原型"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    try:
        tif = ida_typeinf.tinfo_t(prototype, None, ida_typeinf.PT_SIL)
        if not tif.is_func():
            raise IDAError(f"Parsed declaration is not a function type")
        if not ida_typeinf.apply_tinfo(func.start_ea, tif, ida_typeinf.PT_SIL):
            raise IDAError(f"Failed to apply type")
        refresh_decompiler_ctext(func.start_ea)
    except Exception as e:
        raise IDAError(f"Failed to parse prototype string: {prototype}")
    # 自动记录变更
    record_incremental_change("set_function_prototype", {"address": function_address, "prototype": prototype})

class my_modifier_t(ida_hexrays.user_lvar_modifier_t):
    def __init__(self, var_name: str, new_type: ida_typeinf.tinfo_t):
        ida_hexrays.user_lvar_modifier_t.__init__(self)
        self.var_name = var_name
        self.new_type = new_type

    def modify_lvars(self, lvars):
        for lvar_saved in lvars.lvvec:
            lvar_saved: ida_hexrays.lvar_saved_info_t
            if lvar_saved.name == self.var_name:
                lvar_saved.type = self.new_type
                return True
        return False

# 注意：这是一种非常不规范的方法，但为了从IDA中获取错误信息是必要的
def parse_decls_ctypes(decls: str, hti_flags: int) -> tuple[int, str]:
    if sys.platform == "win32":
        import ctypes

        assert isinstance(decls, str), "decls must be a string"
        assert isinstance(hti_flags, int), "hti_flags must be an int"
        c_decls = decls.encode("utf-8")
        c_til = None
        ida_dll = ctypes.CDLL("ida")
        ida_dll.parse_decls.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        ida_dll.parse_decls.restype = ctypes.c_int

        messages = []

        @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
        def magic_printer(fmt: bytes, arg1: bytes):
            if fmt.count(b"%") == 1 and b"%s" in fmt:
                formatted = fmt.replace(b"%s", arg1)
                messages.append(formatted.decode("utf-8"))
                return len(formatted) + 1
            else:
                messages.append(f"unsupported magic_printer fmt: {repr(fmt)}")
                return 0

        errors = ida_dll.parse_decls(c_til, c_decls, magic_printer, hti_flags)
    else:
        # 注意：上面的方法也可以在其他平台上工作，但未经过测试，并且存在变量参数ABI的差异。
        errors = ida_typeinf.parse_decls(None, decls, False, hti_flags)
        messages = []
    return errors, messages

@jsonrpc
@idawrite
def declare_c_type(
    c_declaration: Annotated[str, "类型C声明。示例包括：typedef int foo_t; struct bar { int a; bool b; };"],
):
    """从C声明创建或更新本地类型"""
    # PT_SIL: 抑制警告对话框（虽然看起来在这里是不必要的）
    # PT_EMPTY: 允许空类型（也可能是多余的？）
    # PT_TYP: 打印带有结构体标签的状态消息
    flags = ida_typeinf.PT_SIL | ida_typeinf.PT_EMPTY | ida_typeinf.PT_TYP
    errors, messages = parse_decls_ctypes(c_declaration, flags)

    pretty_messages = "\n".join(messages)
    if errors > 0:
        raise IDAError(f"Failed to parse type:\n{c_declaration}\n\nErrors:\n{pretty_messages}")
    return f"success\n\nInfo:\n{pretty_messages}"
    # 自动记录变更
    record_incremental_change("declare_c_type", {"c_declaration": c_declaration})

@jsonrpc
@idawrite
def set_local_variable_type(
    function_address: Annotated[str, "要反编译的函数地址"],
    variable_name: Annotated[str, "变量名称"],
    new_type: Annotated[str, "变量的新类型"],
):
    """设置本地变量的类型"""
    try:
        # 某些版本的 IDA 不支持此构造函数
        new_tif = ida_typeinf.tinfo_t(new_type, None, ida_typeinf.PT_SIL)
    except Exception:
        try:
            new_tif = ida_typeinf.tinfo_t()
            # parse_decl 需要分号来表示类型
            ida_typeinf.parse_decl(new_tif, None, new_type + ";", ida_typeinf.PT_SIL)
        except Exception:
            raise IDAError(f"Failed to parse type: {new_type}")
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, variable_name, variable_name):
        raise IDAError(f"Failed to find local variable: {variable_name}")
    modifier = my_modifier_t(variable_name, new_tif)
    if not ida_hexrays.modify_user_lvars(func.start_ea, modifier):
        raise IDAError(f"Failed to modify local variable: {variable_name}")
    refresh_decompiler_ctext(func.start_ea)
    # 自动记录变更
    record_incremental_change("set_local_variable_type", {"function_address": function_address, "variable_name": variable_name, "new_type": new_type})

class StackFrameVariable(TypedDict):
    name: str
    offset: str
    size: str
    type: str

@jsonrpc
@idaread
def get_stack_frame_variables(
        function_address: Annotated[str, "要获取栈帧变量的反汇编函数地址"]
) -> list[StackFrameVariable]:
    """获取给定函数的栈帧变量"""
    return get_stack_frame_variables_internal(parse_address(function_address))

def get_stack_frame_variables_internal(function_address: int) -> list[dict]:
    func = idaapi.get_func(function_address)
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    members = []
    tif = ida_typeinf.tinfo_t()
    if not tif.get_type_by_tid(func.frame) or not tif.is_udt():
        return []

    udt = ida_typeinf.udt_type_data_t()
    tif.get_udt_details(udt)
    for udm in udt:
        if not udm.is_gap():
            name = udm.name
            offset = udm.offset // 8
            size = udm.size // 8
            type = str(udm.type)

            members += [StackFrameVariable(name=name,
                                           offset=hex(offset),
                                           size=hex(size),
                                           type=type)
            ]

    return members


class StructureMember(TypedDict):
    name: str
    offset: str
    size: str
    type: str

class StructureDefinition(TypedDict):
    name: str
    size: str
    members: list[StructureMember]

@jsonrpc
@idaread
def get_defined_structures() -> list[StructureDefinition]:
    """返回所有定义的结构体列表"""

    rv = []
    limit = ida_typeinf.get_ordinal_limit()
    for ordinal in range(1, limit):
        tif = ida_typeinf.tinfo_t()
        tif.get_numbered_type(None, ordinal)
        if tif.is_udt():
            udt = ida_typeinf.udt_type_data_t()
            members = []
            if tif.get_udt_details(udt):
                members = [
                    StructureMember(name=x.name,
                                    offset=hex(x.offset // 8),
                                    size=hex(x.size // 8),
                                    type=str(x.type))
                    for _, x in enumerate(udt)
                ]

            rv += [StructureDefinition(name=tif.get_type_name(),
                                       size=hex(tif.get_size()),
                                       members=members)]

    return rv

@jsonrpc
@idawrite
def rename_stack_frame_variable(
        function_address: Annotated[str, "要设置栈帧变量的反汇编函数地址"],
        old_name: Annotated[str, "变量的当前名称"],
        new_name: Annotated[str, "变量的新名称 (空表示默认名称)"]
):
    """更改IDA函数中栈帧变量的名称"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(old_name)
    if not udm:
        raise IDAError(f"{old_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    if ida_frame.is_special_frame_member(tid):
        raise IDAError(f"{old_name} is a special frame member. Will not change the name.")

    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8
    if ida_frame.is_funcarg_off(func, offset):
        raise IDAError(f"{old_name} is an argument member. Will not change the name.")

    sval = ida_frame.soff_to_fpoff(func, offset)
    if not ida_frame.define_stkvar(func, new_name, sval, udm.type):
        raise IDAError("failed to rename stack frame variable")

@jsonrpc
@idawrite
def create_stack_frame_variable(
        function_address: Annotated[str, "要设置栈帧变量的反汇编函数地址"],
        offset: Annotated[str, "栈帧变量的偏移量"],
        variable_name: Annotated[str, "栈变量名称"],
        type_name: Annotated[str, "栈变量类型"]
):
    """对于给定的函数，在指定偏移量处创建一个栈变量并设置特定类型"""

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    offset = parse_address(offset)

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    tif = get_type_by_name(type_name)
    if not ida_frame.define_stkvar(func, variable_name, offset, tif):
        raise IDAError("failed to define stack frame variable")

@jsonrpc
@idawrite
def set_stack_frame_variable_type(
        function_address: Annotated[str, "要设置栈帧变量的反汇编函数地址"],
        variable_name: Annotated[str, "栈变量名称"],
        type_name: Annotated[str, "栈变量类型"]
):
    """对于给定的反汇编函数，设置栈变量的类型"""

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(variable_name)
    if not udm:
        raise IDAError(f"{variable_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8

    tif = get_type_by_name(type_name)
    if not ida_frame.set_frame_member_type(func, offset, tif):
        raise IDAError("failed to set stack frame variable type")

@jsonrpc
@idawrite
def delete_stack_frame_variable(
        function_address: Annotated[str, "要设置栈帧变量的函数地址"],
        variable_name: Annotated[str, "栈变量名称"]
):
    """删除给定函数的命名栈变量"""

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(variable_name)
    if not udm:
        raise IDAError(f"{variable_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    if ida_frame.is_special_frame_member(tid):
        raise IDAError(f"{variable_name} is a special frame member. Will not delete.")

    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8
    size = udm.size // 8
    if ida_frame.is_funcarg_off(func, offset):
        raise IDAError(f"{variable_name} is an argument member. Will not delete.")

    if not ida_frame.delete_frame_members(func, offset, offset+size):
        raise IDAError("failed to delete stack frame variable")

@jsonrpc
@idaread
def read_memory_bytes(
        memory_address: Annotated[str, "要读取的字节地址"],
        size: Annotated[int, "要读取的内存大小"]
) -> str:
    """
    读取指定地址的字节。

    仅当 `get_global_variable_at` 和 `get_global_variable_by_name`
    都失败时才使用此函数。
    """
    return ' '.join(f'{x:#02x}' for x in ida_bytes.get_bytes(parse_address(memory_address), size))

@jsonrpc
@idaread
def data_read_byte(
    address: Annotated[str, "要获取 1 字节值的地址"],
) -> int:
    """
    读取指定地址的 1 字节值。

    仅当 `get_global_variable_at` 失败时才使用此函数。
    """
    ea = parse_address(address)
    return ida_bytes.get_wide_byte(ea)

@jsonrpc
@idaread
def data_read_word(
    address: Annotated[str, "要获取 2 字节值的地址"],
) -> int:
    """
    读取指定地址的 2 字节值作为 WORD。

    仅当 `get_global_variable_at` 失败时才使用此函数。
    """
    ea = parse_address(address)
    return ida_bytes.get_wide_word(ea)

@jsonrpc
@idaread
def data_read_dword(
    address: Annotated[str, "要获取 4 字节值的地址"],
) -> int:
    """
    读取指定地址的 4 字节值作为 DWORD。

    仅当 `get_global_variable_at` 失败时才使用此函数。
    """
    ea = parse_address(address)
    return ida_bytes.get_wide_dword(ea)

@jsonrpc
@idaread
def data_read_qword(
        address: Annotated[str, "要获取 8 字节值的地址"]
) -> int:
    """
    读取指定地址的 8 字节值作为 QWORD。

    仅当 `get_global_variable_at` 失败时才使用此函数。
    """
    ea = parse_address(address)
    return ida_bytes.get_qword(ea)

@jsonrpc
@idaread
def data_read_string(
        address: Annotated[str, "要获取字符串的地址"]
) -> str:
    """
    读取指定地址的字符串。

    仅当 `get_global_variable_at` 失败时才使用此函数。
    """
    try:
        return idaapi.get_strlit_contents(parse_address(address),-1,0).decode("utf-8")
    except Exception as e:
        return "Error:" + str(e)

@jsonrpc
@idaread
@unsafe
def dbg_get_registers() -> list[dict[str, str]]:
    """获取所有寄存器及其值。此函数仅在调试时可用。"""
    result = []
    dbg = ida_idd.get_dbg()
    for thread_index in range(ida_dbg.get_thread_qty()):
        tid = ida_dbg.getn_thread(thread_index)
        regs = []
        regvals = ida_dbg.get_reg_vals(tid)
        for reg_index, rv in enumerate(regvals):
            reg_info = dbg.regs(reg_index)
            reg_value = rv.pyval(reg_info.dtype)
            if isinstance(reg_value, int):
                try_record_dynamic_string(reg_value)
                reg_value = hex(reg_value)
            if isinstance(reg_value, bytes):
                reg_value = reg_value.hex(" ")
            regs.append({
                "name": reg_info.name,
                "value": reg_value,
            })
        result.append({
            "thread_id": tid,
            "registers": regs,
        })
    return result

@jsonrpc
@idaread
@unsafe
def dbg_get_call_stack() -> list[dict[str, str]]:
    """获取当前调用堆栈。"""
    callstack = []
    try:
        tid = ida_dbg.get_current_thread()
        trace = ida_idd.call_stack_t()

        if not ida_dbg.collect_stack_trace(tid, trace):
            return []
        for frame in trace:
            frame_info = {
                "address": hex(frame.callea),
            }
            try:
                module_info = ida_idd.modinfo_t()
                if ida_dbg.get_module_info(frame.callea, module_info):
                    frame_info["module"] = os.path.basename(module_info.name)
                else:
                    frame_info["module"] = "<unknown>"

                name = (
                    ida_name.get_nice_colored_name(
                        frame.callea,
                        ida_name.GNCN_NOCOLOR
                        | ida_name.GNCN_NOLABEL
                        | ida_name.GNCN_NOSEG
                        | ida_name.GNCN_PREFDBG,
                    )
                    or "<unnamed>"
                )
                frame_info["symbol"] = name

            except Exception as e:
                frame_info["module"] = "<error>"
                frame_info["symbol"] = str(e)

            callstack.append(frame_info)

    except Exception as e:
        pass
    return callstack

def list_breakpoints():
    ea = ida_ida.inf_get_min_ea()
    end_ea = ida_ida.inf_get_max_ea()
    breakpoints = []
    while ea <= end_ea:
        bpt = ida_dbg.bpt_t()
        if ida_dbg.get_bpt(ea, bpt):
            breakpoints.append(
                {
                    "ea": hex(bpt.ea),
                    "type": bpt.type,
                    "enabled": bpt.flags & ida_dbg.BPT_ENABLED,
                    "condition": bpt.condition if bpt.condition else None,
                }
            )
        ea = ida_bytes.next_head(ea, end_ea)
    return breakpoints

@jsonrpc
@idaread
@unsafe
def dbg_list_breakpoints():
    """列出程序中的所有断点。"""
    return list_breakpoints()

@jsonrpc
@idaread
@unsafe
def dbg_start_process() -> str:
    """启动调试器"""
    if idaapi.start_process("", "", ""):
        return "Debugger started"
    return "Failed to start debugger"

@jsonrpc
@idaread
@unsafe
def dbg_exit_process() -> str:
    """退出调试器"""
    if idaapi.exit_process():
        return "Debugger exited"
    return "Failed to exit debugger"

@jsonrpc
@idaread
@unsafe
def dbg_continue_process() -> str:
    """继续调试器"""
    if idaapi.continue_process():
        return "Debugger continued"
    return "Failed to continue debugger"

@jsonrpc
@idaread
@unsafe
def dbg_run_to(
    address: Annotated[str, "运行调试器到指定地址"],
) -> str:
    """运行调试器到指定地址"""
    ea = parse_address(address)
    if idaapi.run_to(ea):
        return f"Debugger run to {hex(ea)}"
    return f"Failed to run to address {hex(ea)}"

@jsonrpc
@idaread
@unsafe
def dbg_set_breakpoint(
    address: Annotated[str, "在指定地址设置断点"],
) -> str:
    """在指定地址设置断点"""
    ea = parse_address(address)
    if idaapi.add_bpt(ea, 0, idaapi.BPT_SOFT):
        return f"Breakpoint set at {hex(ea)}"
    breakpoints = list_breakpoints()
    for bpt in breakpoints:
        if bpt["ea"] == hex(ea):
            return f"Breakpoint already exists at {hex(ea)}"
    return f"Failed to set breakpoint at address {hex(ea)}"

@jsonrpc
@idaread
@unsafe
def dbg_delete_breakpoint(
    address: Annotated[str, "del a breakpoint at the specified address"],
) -> str:
    """del a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.del_bpt(ea):
        return f"Breakpoint deleted at {hex(ea)}"
    return f"Failed to delete breakpoint at address {hex(ea)}"

@jsonrpc
@idaread
@unsafe
def dbg_enable_breakpoint(
    address: Annotated[str, "Enable or disable a breakpoint at the specified address"],
    enable: Annotated[bool, "Enable or disable a breakpoint"],
) -> str:
    """Enable or disable a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.enable_bpt(ea, enable):
        return f"Breakpoint {'enabled' if enable else 'disabled'} at {hex(ea)}"
    return f"Failed to {'' if enable else 'disable '}breakpoint at address {hex(ea)}"

@jsonrpc
@idawrite
def batch_rename_functions(
    renames: Annotated[list[dict], "批量重命名参数列表，每项包含 address 和 new_name"]
) -> list[dict]:
    """
    批量重命名函数。
    参数：
        renames: [{"address": "0x401000", "new_name": "check_password"}, ...]
    返回：
        每项包含 address、新名称、是否成功、错误信息（如有）
    """
    results = []
    for item in renames:
        address = item.get("address")
        new_name = item.get("new_name")
        result = {"address": address, "new_name": new_name, "success": False, "error": None}
        try:
            func = idaapi.get_func(parse_address(address))
            if not func:
                result["error"] = f"未找到函数: {address}"
            elif not idaapi.set_name(func.start_ea, new_name):
                result["error"] = f"重命名失败: {address} -> {new_name}"
            else:
                refresh_decompiler_ctext(func.start_ea)
                result["success"] = True
        except Exception as e:
            result["error"] = str(e)
        results.append(result)
    return results

class MCP(idaapi.plugin_t):
    flags = idaapi.PLUGIN_KEEP
    comment = "MCP Plugin"
    help = "MCP"
    wanted_name = "MCP"
    wanted_hotkey = "Ctrl-Alt-M"

    def init(self):
        self.server = Server()
        hotkey = MCP.wanted_hotkey.replace("-", "+")
        if sys.platform == "darwin":
            hotkey = hotkey.replace("Alt", "Option")
        print(f"[MCP] Plugin loaded, use Edit -> Plugins -> MCP ({hotkey}) to start the server")
        return idaapi.PLUGIN_KEEP

    def run(self, args):
        self.server.start()

    def term(self):
        self.server.stop()

def PLUGIN_ENTRY():
    return MCP()

@jsonrpc
@idaread
def get_function_call_graph(
    start_address: Annotated[str, "起始函数地址"],
    depth: Annotated[int, "递归深度"] = 3,
    mermaid: Annotated[bool, "是否返回 mermaid 格式"] = False
) -> dict:
    """
    获取函数调用图。
    参数：
        start_address: 起始函数地址（字符串）
        depth: 递归深度，默认3
        mermaid: 是否返回 mermaid 格式（默认False，返回邻接表）
    返回：
        {"graph": 邻接表或mermaid字符串, "nodes": 节点列表, "edges": 边列表}
    """
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
    start_addr = hex(parse_address(start_address))
    dfs(start_addr, depth)
    nodes = list(nodes)
    edges = list(edges)
    if mermaid:
        mermaid_lines = ["graph TD"]
        for src, dst in edges:
            mermaid_lines.append(f'    "{src}" --> "{dst}"')
        return {"graph": "\n".join(mermaid_lines), "nodes": nodes, "edges": edges}
    else:
        adj = {n: [] for n in nodes}
        for src, dst in edges:
            adj[src].append(dst)
        return {"graph": adj, "nodes": nodes, "edges": edges}

@jsonrpc
@idaread
def detect_obfuscation(
    function_address: Annotated[str, "要检测混淆的函数地址"]
) -> dict:
    """
    检测常见混淆模式，包括：
    - 控制流平坦化（大量switch/case、间接跳转、异常块等）
    - 字符串加密（动态解密、字符串表、引用方式）
    - 反调试（API、PEB、TLS、int 0x2d、异常处理等）
    - 垃圾代码/死代码（无用块、异常分支、无效循环）
    - 代码块分布异常
    返回：
        {"flattening": bool, "string_encryption": bool, "anti_debug": bool, "dead_code": bool, "details": str}
    """
    addr = parse_address(function_address)
    func = idaapi.get_func(addr)
    if not func:
        return {"flattening": False, "string_encryption": False, "anti_debug": False, "dead_code": False, "details": "未找到函数"}
    flowchart = list(ida_gdl.FlowChart(func))
    block_count = len(flowchart)
    branch_count = 0
    switch_count = 0
    indirect_jmp = 0
    dead_code_blocks = 0
    for block in flowchart:
        branch_count += sum(1 for _ in block.succs())
        # 检查是否有switch/case
        for head in idautils.Heads(block.start_ea, block.end_ea):
            mnem = idc.print_insn_mnem(head)
            if mnem in ("jmp", "call") and idc.get_operand_type(head, 0) == idc.o_mem:
                indirect_jmp += 1
            if mnem == "jmp" and idc.get_operand_type(head, 0) == idc.o_phrase:
                switch_count += 1
    flattening = (block_count > 10 and branch_count / block_count < 1.5) or switch_count > 2 or indirect_jmp > 2
    # 死代码检测：无前驱的块
    entry_ea = func.start_ea
    reachable = set()
    def dfs(ea):
        if ea in reachable:
            return
        reachable.add(ea)
        for succ in idautils.CodeRefsFrom(ea, 0):
            if func.start_ea <= succ < func.end_ea:
                dfs(succ)
    dfs(entry_ea)
    for block in flowchart:
        if block.start_ea not in reachable:
            dead_code_blocks += 1
    dead_code = dead_code_blocks > 0
    # 字符串加密检测
    string_refs = 0
    dynamic_string_patterns = 0
    for head in idautils.Heads(func.start_ea, func.end_ea):
        mnem = idc.print_insn_mnem(head)
        if mnem in ("mov", "lea"):
            for op in range(2):
                if idc.get_operand_type(head, op) == idc.o_mem:
                    ea = idc.get_operand_value(head, op)
                    if ida_bytes.is_strlit(ida_bytes.get_full_flags(ea)):
                        string_refs += 1
        # 检查是否有动态解密/解码调用
        if mnem in ("call", "jmp"):
            callee = idc.get_operand_value(head, 0)
            name = idc.get_name(callee)
            if name and any(x in name.lower() for x in ["decrypt", "decode", "crypt", "rc4", "base64", "tea"]):
                dynamic_string_patterns += 1
    string_encryption = string_refs < 2 or dynamic_string_patterns > 0
    # 反调试检测
    anti_debug = False
    anti_keywords = ["isdebuggerpresent", "checkremotedebuggerpresent", "tls", "int 0x2d", "peb", "beingdebugged", "ntglobalflag", "heapflags", "outputdebugstring", "findwindow", "gettickcount", "rdtsc", "queryperformancecounter"]
    code = decompile_function(hex(addr)).lower()
    for ak in anti_keywords:
        if ak in code:
            anti_debug = True
            break
    # 详细信息
    details = f"基本块数: {block_count}, 分支数: {branch_count}, switch: {switch_count}, 间接跳转: {indirect_jmp}, 死代码块: {dead_code_blocks}, 字符串引用: {string_refs}, 动态字符串模式: {dynamic_string_patterns}, 反调试: {anti_debug}"
    return {"flattening": flattening, "string_encryption": string_encryption, "anti_debug": anti_debug, "dead_code": dead_code, "details": details}

@jsonrpc
@idaread
def get_algorithm_signature(
    function_address: Annotated[str, "要识别算法的函数地址"]
) -> dict:
    """
    自动识别常见算法（如md5、base64、aes、des、sha1、sha256、crc32、tea、xor、zlib、lzma、rot13等），返回算法类型和置信度。
    """
    addr = parse_address(function_address)
    code = decompile_function(hex(addr)).lower()
    # 关键词和特征表
    algo_patterns = [
        ("md5", ["md5", "ff 31", "d41d8cd9", "md5init", "md5update", "md5final"]),
        ("sha1", ["sha1", "67452301efcdab8998badcfe10325476c3d2e1f0", "sha1init", "sha1update", "sha1final"]),
        ("sha256", ["sha256", "6a09e667", "bb67ae85", "3c6ef372", "a54ff53a", "510e527f", "9b05688c", "1f83d9ab", "sha256init", "sha256update", "sha256final"]),
        ("crc32", ["crc32", "edb88320", "04c11db7"]),
        ("aes", ["aes", "rijndael", "sbox", "inv_sbox", "0x63", "0x1b", "0x11b", "aes_encrypt", "aes_decrypt"]),
        ("des", ["des", "0x133457799bbcdff1", "des_encrypt", "des_decrypt"]),
        ("rc4", ["rc4", "sbox", "ksa", "prga", "rc4init"]),
        ("tea", ["tea", "delta=0x9e3779b9", "sum=0xc6ef3720", "tea_encrypt", "tea_decrypt"]),
        ("xor", ["xor", "^", "xor_encrypt", "xor_decrypt"]),
        ("base64", ["base64", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="]),
        ("base32", ["base32", "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567="]),
        ("base58", ["base58", "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"]),
        ("base85", ["base85", "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwx"]),
        ("rot13", ["rot13", "abcdefghijklmnopqrstuvwxyz", "nopqrstuvwxyzabcdefghijklm"]),
        ("zlib", ["zlib", "deflate", "inflate", "adler32"]),
        ("lzma", ["lzma", "xz", "lzma_stream"]),
        ("mersenne_twister", ["mersenne", "mt19937", "0x9908b0df"]),
    ]
    found = []
    for algo, patterns in algo_patterns:
        for pat in patterns:
            if pat in code:
                found.append(algo)
                break
    if found:
        # 置信度: 匹配数量/总数
        confidence = min(1.0, len(found)/3)
        return {"algorithm": ", ".join(found), "confidence": confidence}
    else:
        return {"algorithm": "unknown", "confidence": 0.0}

@jsonrpc
@idaread
def get_patch_points(
    function_address: Annotated[str, "要定位patch点的函数地址"]
) -> list:
    """
    自动定位可 patch 位置（如条件跳转、反调试点）。
    """
    addr = parse_address(function_address)
    func = idaapi.get_func(addr)
    if not func:
        return []
    patch_points = []
    for head in idautils.Heads(func.start_ea, func.end_ea):
        mnem = idc.print_insn_mnem(head)
        if mnem in ("jz", "jnz", "je", "jne", "call"):
            patch_points.append({"address": hex(head), "mnem": mnem})
        # 反调试点示例
        if mnem == "int" and idc.get_operand_value(head, 0) == 0x2d:
            patch_points.append({"address": hex(head), "mnem": "anti-debug"})
    return patch_points

@jsonrpc
@idaread
def get_analysis_report() -> dict:
    """
    自动生成结构化分析报告。
    """
    report = {
        "functions": [],
        "globals": [],
        "strings": [],
        "entry_points": [],
    }
    for f in idautils.Functions():
        func = get_function(f, raise_error=False)
        if func:
            report["functions"].append(func)
    for g in idautils.Names():
        if not idaapi.get_func(g[0]):
            report["globals"].append({"address": hex(g[0]), "name": g[1]})
    for s in idautils.Strings():
        report["strings"].append({"address": hex(s.ea), "string": str(s)})
    report["entry_points"] = get_entry_points()
    return report

@jsonrpc
@idaread
def get_incremental_changes() -> list:
    """
    返回自上次分析以来的增量变更。
    """
    global _incremental_changes
    changes = _incremental_changes.copy()
    _incremental_changes.clear()
    return changes

@jsonrpc
@idaread
def get_dynamic_string_map() -> dict:
    """
    动态字符串解密映射（静态+动态分析结果）。
    """
    string_map = {}
    for s in idautils.Strings():
        string_map[hex(s.ea)] = str(s)
    # 合并动态字符串
    string_map.update(_dynamic_strings)
    return string_map

@jsonrpc
@idaread
def generate_analysis_report_md() -> str:
    """
    一键生成结构化 markdown 报告，帮助用户快速理解程序核心逻辑。
    """
    import hashlib
    # 基本信息
    meta = get_metadata()
    md = [f"# 程序自动分析报告\n"]
    md.append(f"## 基本信息\n- 文件名: {meta['module']}\n- MD5: {meta['md5']}\n- 入口点: {meta['base']}\n")

    # 入口点分析
    entry_points = get_entry_points()
    md.append(f"## 入口点分析\n- 入口点数量: {len(entry_points)}\n" + "\n".join([f"- {f['name']} @ {f['address']}" for f in entry_points]))

    # 导入表分析
    suspicious_apis = ["virtualalloc", "getprocaddress", "loadlibrary", "system", "exec", "winexec", "createthread", "writeprocessmemory", "readprocessmemory", "openprocess", "socket", "connect", "recv", "send"]
    imports = []
    suspicious_imports = []
    for i in range(0, 1000, 100):
        page = list_imports(i, 100)
        for imp in page["data"]:
            imports.append(f"- {imp['imported_name']} ({imp['module']}) @ {imp['address']}")
            if any(api in imp['imported_name'].lower() for api in suspicious_apis):
                suspicious_imports.append(f"- {imp['imported_name']} ({imp['module']}) @ {imp['address']}")
        if not page["next_offset"]:
            break
    md.append(f"\n## 导入表分析\n- 导入API总数: {len(imports)}\n- 可疑API: {len(suspicious_imports)}\n" + ("\n".join(suspicious_imports) if suspicious_imports else "无"))

    # 关键/可疑函数
    keywords = ["flag", "ctf", "check", "verify", "rc4", "base64", "tea", "debug", "tls", "anti", "success", "congrat"]
    suspicious_funcs = []
    algo_funcs = []
    anti_debug_funcs = []
    obfuscated_funcs = []
    func_lens = []
    branch_counts = []
    for f in idautils.Functions():
        func = get_function(f, raise_error=False)
        if not func:
            continue
        name = func["name"].lower()
        code = decompile_function(func["address"])
        func_len = int(func["size"], 16)
        func_lens.append(func_len)
        # 统计分支数
        try:
            flowchart = list(ida_gdl.FlowChart(idaapi.get_func(f)))
            branch_count = sum(len(list(block.succs())) for block in flowchart)
            branch_counts.append(branch_count)
        except:
            branch_counts.append(0)
        for kw in keywords:
            if kw in name or kw in code.lower():
                suspicious_funcs.append(f"- {func['name']} ({func['address']})")
                break
        # 算法检测（升级：展示所有检测到的算法和置信度）
        algo_info = get_algorithm_signature(func["address"])
        if algo_info["algorithm"] != "unknown":
            algo_funcs.append(f"- {func['name']} ({func['address']}) : {algo_info['algorithm']} (置信度: {algo_info['confidence']:.2f})")
        # 反调试检测
        anti_keywords = ["isdebuggerpresent", "checkremotedebuggerpresent", "tls", "int 0x2d", "peb", "beingdebugged"]
        if any(ak in code.lower() for ak in anti_keywords):
            anti_debug_funcs.append(f"- {func['name']} ({func['address']})")
        # 混淆检测
        obf = detect_obfuscation(func["address"])
        if obf.get("flattening") or obf.get("string_encryption"):
            obfuscated_funcs.append(f"- {func['name']} ({func['address']}) : {obf}")
    md.append("\n## 关键/可疑函数\n" + ("\n".join(suspicious_funcs) if suspicious_funcs else "无"))
    md.append("\n## 检测到的加密/编码/哈希算法\n" + ("\n".join(algo_funcs) if algo_funcs else "无"))
    md.append("\n## 反调试相关函数\n" + ("\n".join(anti_debug_funcs) if anti_debug_funcs else "无"))
    md.append("\n## 检测到的混淆/加密函数\n" + ("\n".join(obfuscated_funcs) if obfuscated_funcs else "无"))

    # 关键字符串
    suspicious_strs = []
    for s in idautils.Strings():
        sval = str(s).lower()
        if any(kw in sval for kw in keywords):
            suspicious_strs.append(f"- {sval} @ {hex(s.ea)}")
    md.append("\n## 关键字符串\n" + ("\n".join(suspicious_strs) if suspicious_strs else "无"))

    # flag 逻辑与长度
    flag_info = []
    for f in idautils.Functions():
        func = get_function(f, raise_error=False)
        if not func:
            continue
        code = decompile_function(func["address"])
        if any(kw in code.lower() for kw in ["flag", "ctf", "check", "verify"]):
            constraints = get_function_constraints(func["address"])
            if constraints:
                flag_info.append(f"- {func['name']} ({func['address']}): {constraints}")
    md.append("\n## flag 逻辑与长度\n" + ("\n".join(flag_info) if flag_info else "无"))

    # 代码段/数据段分布
    segs = []
    for seg in idaapi.get_segm_qty() and [idaapi.getnseg(i) for i in range(idaapi.get_segm_qty())]:
        segs.append(f"- {idaapi.get_segm_name(seg)}: {hex(seg.start_ea)} ~ {hex(seg.end_ea)} (大小: {hex(seg.end_ea - seg.start_ea)}) 类型: {seg.type}")
    md.append("\n## 代码段/数据段分布\n" + ("\n".join(segs) if segs else "无"))

    # 代码复杂度
    if func_lens:
        md.append(f"\n## 代码复杂度\n- 函数总数: {len(func_lens)}\n- 平均函数长度: {sum(func_lens)//len(func_lens)} 字节\n- 最大函数长度: {max(func_lens)} 字节\n- 最小函数长度: {min(func_lens)} 字节\n- 平均分支数: {sum(branch_counts)//len(branch_counts) if branch_counts else 0}\n")
    else:
        md.append("\n## 代码复杂度\n无")

    # 交叉引用热点
    xref_func_count = {}
    for f in idautils.Functions():
        xrefs = get_xrefs_to(hex(f))
        xref_func_count[f] = len(xrefs)
    top_funcs = sorted(xref_func_count.items(), key=lambda x: x[1], reverse=True)[:5]
    md.append("\n## 交叉引用热点（函数）\n" + "\n".join([f"- {get_function(f, raise_error=False)['name']} ({hex(f)}): {cnt} 处引用" for f, cnt in top_funcs]))

    # 结构体/类型定义
    structs = get_defined_structures()
    md.append(f"\n## 结构体/类型定义\n- 总数: {len(structs)}\n" + ("\n".join([f"- {s['name']} (大小: {s['size']})" for s in structs[:5]]) if structs else "无"))

    # 主执行流程图（入口点递归3层）
    if entry_points:
        entry_addr = entry_points[0]["address"]
        call_graph = get_function_call_graph(entry_addr, 3, True)
        md.append("\n## 主执行流程图\n```mermaid\n" + call_graph["graph"] + "\n```")
    else:
        md.append("\n## 主执行流程图\n无入口点")

    # 其它自动分析结论
    md.append("\n## 其它自动分析结论\n")
    # 反调试点补充
    anti_debug_points = []
    for f in idautils.Functions():
        func = get_function(f, raise_error=False)
        if not func:
            continue
        patch_points = get_patch_points(func["address"])
        for pt in patch_points:
            if pt["mnem"] in ("anti-debug", "int", "tls"):
                anti_debug_points.append(f"- {func['name']} {pt['address']} : {pt['mnem']}")
    if anti_debug_points:
        md.append("### 反调试点\n" + "\n".join(anti_debug_points))
    else:
        md.append("### 反调试点\n无")
    # 未命名函数比例
    unnamed = [f for f in idautils.Functions() if get_function(f, raise_error=False) and get_function(f, raise_error=False)["name"].startswith("sub_")]
    md.append(f"\n- 未命名函数数量: {len(unnamed)} / {len(func_lens)}\n")
    return "\n".join(md)
