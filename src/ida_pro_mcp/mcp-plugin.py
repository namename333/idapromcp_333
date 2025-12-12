import os
import sys
import traceback

# 修改Python版本要求，使其兼容IDA Pro环境
# IDA Pro 通常使用Python 3.8或3.9
if sys.version_info < (3, 8):
    print("[MCP] 警告: 建议使用Python 3.8或更高版本以获得最佳体验")
    # 不再直接抛出异常，而是给出警告并继续执行

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

# 导入脚本生成工具模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import script_utils
except ImportError:
    print("[MCP] 警告: 无法导入script_utils模块，将使用内置函数")
    script_utils = None

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

@jsonrpc
@idaread
def check_connection() -> dict:
    """
    标准MCP协议接口：检查与服务器的连接
    用于客户端验证服务是否正常运行
    """
    return {
        "status": "ok",
        "protocol": "MCP",
        "version": "1.6.0",
        "server": "IDA Pro MCP Plugin",
        "timestamp": time.time()
    }

@jsonrpc
@idaread
def get_methods() -> list[dict]:
    """
    获取所有可用的JSON-RPC方法列表及其元数据
    支持MCP协议的自描述功能
    """
    methods_info = []
    for method_name, func in rpc_registry.methods.items():
        method_info = {
            "name": method_name,
            "description": func.__doc__ or "",
            "is_unsafe": method_name in rpc_registry.unsafe,
            "parameters": []
        }
        
        # 尝试获取函数参数信息
        try:
            import inspect
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "unknown"
                }
                method_info["parameters"].append(param_info)
        except:
            # 如果获取参数信息失败，继续处理其他方法
            pass
        
        methods_info.append(method_info)
    
    return methods_info

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
        logger.info(f"收到POST请求: {self.path}")

        parsed_path = urlparse(self.path)
        # 同时支持/jsonrpc和/mcp路径，确保兼容性
        if parsed_path.path not in ["/jsonrpc", "/mcp"]:
            logger.error(f"无效的接口路径: {parsed_path.path}")
            self.send_jsonrpc_error(-32098, "无效的接口路径", None)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            logger.error("请求体缺失")
            self.send_jsonrpc_error(-32700, "请求体缺失", None)
            return

        request_body = self.rfile.read(content_length)
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            logger.error(f"JSON 解析错误: {request_body}")
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
                logger.error(f"请求格式错误: {request}")
                raise JSONRPCError(-32600, "请求格式错误")
            if request.get("jsonrpc") != "2.0":
                logger.error(f"JSON-RPC 版本错误: {request.get('jsonrpc')}")
                raise JSONRPCError(-32600, "JSON-RPC 版本错误")
            if "method" not in request:
                logger.error("未指定方法名")
                raise JSONRPCError(-32600, "未指定方法名")

            method = request["method"]
            params = request.get("params", [])
            logger.info(f"处理API请求: {method}, 参数: {params}")
            
            # 分发方法调用
            result = rpc_registry.dispatch(method, params)
            response["result"] = result
            logger.info(f"API请求成功完成: {method}")

        except JSONRPCError as e:
            logger.error(f"JSONRPC错误: {e.code} - {e.message}")
            response["error"] = {
                "code": e.code,
                "message": e.message
            }
            if e.data is not None:
                response["error"]["data"] = e.data
        except IDAError as e:
            logger.error(f"IDA错误: {e.message}")
            response["error"] = {
                "code": -32000,
                "message": e.message,
            }
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"内部错误: {str(e)}\n{error_trace}")
            response["error"] = {
                "code": -32603,
                "message": "内部错误（请反馈 bug）",
                "data": error_trace,
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

def get_config_file_path():
    """
    获取配置文件路径
    按优先级查找配置文件：
    1. 当前工作目录下的mcp_config.json
    2. 用户目录下的.mcp/mcp_config.json
    3. IDA插件目录下的mcp_config.json
    """
    # 尝试多个配置文件位置，按优先级返回第一个存在的
    import ida_idaapi
    import pathlib
    
    # 获取可能的配置文件路径列表
    config_paths = []
    
    # 1. 当前工作目录
    config_paths.append(os.path.join(os.getcwd(), "mcp_config.json"))
    
    # 2. 用户目录下的.mcp文件夹
    user_home = os.path.expanduser("~")
    config_paths.append(os.path.join(user_home, ".mcp", "mcp_config.json"))
    
    # 3. IDA插件目录
    plugin_dir = ida_idaapi.idadir("plugins")
    config_paths.append(os.path.join(plugin_dir, "mcp_config.json"))
    
    # 返回第一个存在的配置文件
    for path in config_paths:
        if os.path.exists(path):
            logger.info(f"使用配置文件: {path}")
            return path
    
    # 如果都不存在，返回默认路径（IDA插件目录）
    default_path = os.path.join(plugin_dir, "mcp_config.json")
    logger.info(f"未找到配置文件，将使用默认配置。默认配置文件路径: {default_path}")
    return default_path

def validate_config(config):
    """
    验证配置的有效性
    返回(是否有效, 错误信息)
    """
    # 检查必需的配置项
    required_fields = ["host", "port", "allow_port_override"]
    for field in required_fields:
        if field not in config:
            return False, f"缺少必需的配置项: {field}"
    
    # 验证端口号
    port = config.get("port")
    if not isinstance(port, int) or port < 1 or port > 65535:
        return False, f"无效的端口号: {port}"
    
    # 验证主机名
    host = config.get("host")
    if not isinstance(host, str) or not host:
        return False, "主机名不能为空"
    
    # 验证布尔类型配置
    if not isinstance(config.get("allow_port_override"), bool):
        return False, "allow_port_override 必须是布尔值"
    
    return True, ""

def load_config():
    """
    加载配置文件
    返回验证后的配置字典，如果配置无效则返回默认配置
    """
    default_config = {
        "host": "localhost",
        "port": 13337,
        "allow_port_override": True,
        "log_level": "INFO",
        "timeout": 30
    }
    
    config_path = get_config_file_path()
    user_config = {}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            # 合并默认配置和用户配置
            merged_config = {**default_config, **user_config}
            
            # 验证配置
            is_valid, error_msg = validate_config(merged_config)
            if not is_valid:
                logger.error(f"配置验证失败: {error_msg}，将使用默认配置")
                return default_config
            
            logger.info(f"成功加载并验证配置文件: {config_path}")
            return merged_config
            
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}，将使用默认配置")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，将使用默认配置")
    
    # 从环境变量覆盖配置
    if "MCP_PORT" in os.environ:
        try:
            port = int(os.environ["MCP_PORT"])
            if 1 <= port <= 65535:
                default_config["port"] = port
                logger.info(f"从环境变量MCP_PORT设置端口: {port}")
            else:
                logger.warning(f"环境变量MCP_PORT值无效: {port}，必须在1-65535之间")
        except ValueError:
            logger.warning("环境变量MCP_PORT不是有效的整数")
    
    if "MCP_HOST" in os.environ:
        default_config["host"] = os.environ["MCP_HOST"]
        logger.info(f"从环境变量MCP_HOST设置主机: {default_config['host']}")
    
    return default_config

# MCP HTTP 服务器
class MCPHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True  # 允许端口重用

# 服务器主类
class Server:
    def __init__(self):
        self.server = None
        self.server_thread = None
        self.running = False
        # 从配置系统获取配置
        self.config = load_config()
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 13337)
        
    def start(self):
        """
        启动MCP服务器
        初始化并启动一个线程来运行服务器
        """
        try:
            if self.running:
                print("[MCP] 服务器已在运行")
                logger.info("服务器已在运行")
                return

            # 确保之前的资源已释放
            if self.server is not None:
                try:
                    self.server.server_close()
                except:
                    pass
                self.server = None
            
            if self.server_thread is not None:
                try:
                    if self.server_thread.is_alive():
                        logger.warning("检测到之前的服务器线程仍在运行，尝试停止")
                except:
                    pass
                self.server_thread = None

            # 加载配置
            try:
                config = load_config()
                self.host = config["host"]
                self.port = config["port"]
                self.allow_port_override = config.get("allow_port_override", True)
                logger.info(f"加载配置完成 - 主机: {self.host}, 端口: {self.port}")
            except Exception as e:
                print(f"[MCP] 加载配置失败: {e}")
                logger.error(f"加载配置失败: {e}", exc_info=True)
                # 使用默认配置继续
                self.host = "localhost"
                self.port = 13337
                self.allow_port_override = True
                print("[MCP] 使用默认配置继续")

            # 创建并启动服务器线程
            self.server_thread = threading.Thread(target=self._run_server, daemon=True, name="MCP-Server-Thread")
            self.running = True
            self.server_thread.start()
            print("[MCP] 服务器启动中...")
            logger.info("服务器线程已启动")
            
            # 添加短暂延迟确保服务器有机会绑定端口
            time.sleep(0.5)
            
        except Exception as e:
            self.running = False
            print(f"[MCP] 服务器启动失败: {e}")
            logger.error(f"服务器启动失败: {e}", exc_info=True)

    def stop(self):
        """
        停止MCP服务器
        安全地关闭服务器并释放资源
        """
        try:
            if not self.running:
                print("[MCP] 服务器未运行")
                return

            print("[MCP] 正在停止服务器...")
            logger.info("正在停止服务器...")
            
            # 首先标记为非运行状态
            self.running = False
            
            # 优雅关闭服务器
            if self.server:
                try:
                    self.server.shutdown()
                    logger.info("服务器已关闭")
                except Exception as e:
                    logger.warning(f"关闭服务器时出错: {e}")
                
                try:
                    self.server.server_close()
                    logger.info("服务器资源已释放")
                except Exception as e:
                    logger.warning(f"释放服务器资源时出错: {e}")
                
                self.server = None
            
            # 等待线程结束
            if self.server_thread:
                try:
                    # 设置超时以避免无限等待
                    self.server_thread.join(timeout=5.0)
                    if self.server_thread.is_alive():
                        logger.warning("服务器线程未能在超时时间内结束")
                    else:
                        logger.info("服务器线程已结束")
                except Exception as e:
                    logger.warning(f"等待服务器线程结束时出错: {e}")
                
                self.server_thread = None
            
            print("[MCP] 服务器已成功停止")
            logger.info("服务器已成功停止")
            
        except Exception as e:
            print(f"[MCP] 停止服务器时出错: {e}")
            logger.error(f"停止服务器时出错: {e}", exc_info=True)
            # 确保状态被重置
            self.running = False
            self.server = None
            self.server_thread = None

    def _run_server(self):
        """
        运行MCP服务器
        尝试绑定指定端口，如果失败则尝试自动选择可用端口
        """
        try:
            # 尝试绑定指定端口，如果失败则尝试自动选择可用端口
            original_port = self.port  # 保存原始端口
            current_port = original_port
            max_attempts = 100  # 最多尝试100个端口
            attempts = 0
            
            while attempts < max_attempts and self.running:
                try:
                    self.server = MCPHTTPServer((self.host, current_port), JSONRPCRequestHandler)
                    self.port = current_port  # 更新实际使用的端口
                    print(f"[MCP] 服务器已启动: http://{self.host}:{self.port}")
                    logger.info(f"服务器已启动: http://{self.host}:{self.port}")
                    
                    # 如果使用的端口不是原始端口，保存到配置中
                    if current_port != original_port and self.allow_port_override:
                        self._save_port_config(current_port)
                    
                    # 启动服务器
                    self.server.serve_forever()
                    break  # 成功启动后跳出循环
                    
                except OSError as e:
                    # 检查是否是端口被占用错误
                    if "[WinError 10048]" in str(e) or "Address already in use" in str(e):
                        if self.allow_port_override and self.running:
                            attempts += 1
                            current_port += 1
                            logger.warning(f"端口 {current_port-1} 已被占用，尝试使用端口 {current_port}")
                            time.sleep(0.1)  # 短暂延迟，避免过快尝试
                        else:
                            print(f"[MCP] 启动服务器失败: {e}")
                            logger.error(f"启动服务器失败: {e}")
                            break
                    else:
                        print(f"[MCP] 启动服务器失败: {e}")
                        logger.error(f"启动服务器失败: {e}", exc_info=True)
                        break
            
            if attempts >= max_attempts and self.running:
                error_msg = f"[MCP] 启动服务器失败: 无法找到可用端口 (尝试了{max_attempts}次)"
                print(error_msg)
                logger.error(error_msg)
                
        except Exception as e:
            print(f"[MCP] 服务器运行错误: {e}")
            logger.error(f"服务器运行错误: {e}", exc_info=True)
        finally:
            # 确保在退出时更新运行状态
            if self.running:
                self.running = False
                print("[MCP] 服务器已停止运行")
                logger.info("服务器已停止运行")
            
    def _save_port_config(self, port):
        """
        保存端口配置到配置文件
        """
        try:
            config_path = get_config_file_path()
            # 加载现有配置
            config = load_config()
            config["port"] = port
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            logger.info(f"已保存端口配置: {port}")
        except Exception as e:
            logger.warning(f"保存端口配置失败: {e}")
            # 即使保存配置失败，也不应该重复启动服务器

# 一个帮助编写线程安全IDA代码的模块。
# Based on:
# https://web.archive.org/web/20160305190440/http://www.williballenthin.com/blog/2015/09/04/idapython-synchronization-decorator/
import os
import queue
import traceback
import functools
import logging.handlers

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

def setup_logging():
    """
    设置日志系统，支持配置化级别和日志滚动
    """
    # 获取日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 从配置加载日志级别
    try:
        config = load_config()
        log_level = getattr(logging, config.get('log_level', 'INFO'))
    except (NameError, AttributeError):
        log_level = logging.INFO
    
    # 创建logger对象
    logger = logging.getLogger('MCP')
    logger.setLevel(log_level)
    
    # 清除已有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建文件handler（支持日志滚动，最多保存5个文件，每个5MB）
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'ida_mcp_plugin.log'),
        maxBytes=5*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(log_level)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 初始化日志系统
logger = setup_logging()

# 安全模式枚举，数值越高表示越安全：
class IDASafety:
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
    """列出数据库中的所有函数（分页）
    
    警告: 此API已弃用，请使用更通用的query_database API
    """
    functions = [get_function(address) for address in idautils.Functions()]
    return paginate(functions, offset, count)

@jsonrpc
@idaread
def query_database(
    entity_type: Annotated[str, "实体类型: 'functions', 'globals', 'strings', 'imports'"],
    offset: Annotated[int, "从 (0) 开始列出偏移量"],
    count: Annotated[int, "要列出的实体数量 (100 是默认值，0 表示剩余)"],
    filter: Annotated[str, "可选的过滤模式，空字符串表示无过滤"],
) -> Page[dict]:
    """统一的数据库查询API，可查询各种实体类型
    
    用于替代特定的list_xxx和list_xxx_filter函数
    """
    if entity_type == 'functions':
        functions = [get_function(address) for address in idautils.Functions()]
        if filter:
            functions = pattern_filter(functions, filter, "name")
        return paginate(functions, offset, count)
    elif entity_type == 'globals':
        return list_globals_filter(offset, count, filter)
    elif entity_type == 'strings':
        return list_strings_filter(offset, count, filter)
    elif entity_type == 'imports':
        imports = []
        for i in idautils.Imports():
            for name, ordinal in idautils.Entries(i):
                if name:
                    imports.append({
                        "address": hex(ordinal),
                        "imported_name": name,
                        "module": idaapi.get_import_module_name(i),
                    })
        if filter:
            imports = pattern_filter(imports, filter, "imported_name")
        return paginate(imports, offset, count)
    else:
        raise JSONRPCError(400, f"不支持的实体类型: {entity_type}")

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
    """列出数据库中的所有全局变量（分页）
    
    警告: 此API已弃用，请使用更通用的query_database API
    """
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
    """列出数据库中的所有字符串（分页）
    
    警告: 此API已弃用，请使用更通用的query_database API
    """
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
        func_name = idc.get_func_name(func.start_ea)
        if label and label == func_name and address == func.start_ea:
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
def dbg_control_process(
    action: Annotated[str, "调试操作类型: 'start', 'exit', 'continue', 'run_to'"],
    address: Annotated[Optional[str], "目标地址，仅run_to操作需要"] = None
) -> str:
    """统一的调试器控制接口
    
    Args:
        action: 调试操作类型，支持'start', 'exit', 'continue', 'run_to'
        address: 目标地址，仅在action为'run_to'时需要
    
    Returns:
        操作结果消息
    """
    if action == 'start':
        if idaapi.start_process("", "", ""):
            return "Debugger started"
        return "Failed to start debugger"
    elif action == 'exit':
        if idaapi.exit_process():
            return "Debugger exited"
        return "Failed to exit debugger"
    elif action == 'continue':
        if idaapi.continue_process():
            return "Debugger continued"
        return "Failed to continue debugger"
    elif action == 'run_to':
        if not address:
            return "Error: Address required for run_to action"
        ea = parse_address(address)
        if idaapi.run_to(ea):
            return f"Debugger run to {hex(ea)}"
        return f"Failed to run to address {hex(ea)}"
    else:
        return f"Error: Invalid action '{action}'. Supported actions: start, exit, continue, run_to"

@jsonrpc
@idaread
@unsafe
def dbg_start_process() -> str:
    """启动调试器 (已弃用，请使用dbg_control_process)"""
    logger.warning("dbg_start_process is deprecated. Please use dbg_control_process with action='start'")
    return dbg_control_process('start')

@jsonrpc
@idaread
@unsafe
def dbg_exit_process() -> str:
    """退出调试器 (已弃用，请使用dbg_control_process)"""
    logger.warning("dbg_exit_process is deprecated. Please use dbg_control_process with action='exit'")
    return dbg_control_process('exit')

@jsonrpc
@idaread
@unsafe
def dbg_continue_process() -> str:
    """继续调试器 (已弃用，请使用dbg_control_process)"""
    logger.warning("dbg_continue_process is deprecated. Please use dbg_control_process with action='continue'")
    return dbg_control_process('continue')

@jsonrpc
@idaread
@unsafe
def dbg_run_to(
    address: Annotated[str, "运行调试器到指定地址"],
) -> str:
    """运行调试器到指定地址 (已弃用，请使用dbg_control_process)"""
    logger.warning("dbg_run_to is deprecated. Please use dbg_control_process with action='run_to'")
    return dbg_control_process('run_to', address)

@jsonrpc
@idaread
@unsafe
def dbg_manage_breakpoint(
    action: Annotated[str, "断点操作类型: 'list', 'set', 'delete', 'enable'"],
    address: Annotated[Optional[str], "断点地址，仅set/delete/enable操作需要"] = None,
    enable: Annotated[Optional[bool], "是否启用断点，仅enable操作需要"] = None
) -> Union[str, list[dict[str, str]]]:
    """统一的断点管理接口
    
    Args:
        action: 断点操作类型，支持'list', 'set', 'delete', 'enable'
        address: 断点地址，仅在action为'set', 'delete', 'enable'时需要
        enable: 是否启用断点，仅在action为'enable'时需要
    
    Returns:
        操作结果消息或断点列表
    """
    if action == 'list':
        return list_breakpoints()
    elif action in ['set', 'delete', 'enable']:
        if not address:
            return f"Error: Address required for {action} action"
        ea = parse_address(address)
        
        if action == 'set':
            if idaapi.add_bpt(ea, 0, idaapi.BPT_SOFT):
                return f"Breakpoint set at {hex(ea)}"
            breakpoints = list_breakpoints()
            for bpt in breakpoints:
                if bpt["ea"] == hex(ea):
                    return f"Breakpoint already exists at {hex(ea)}"
            return f"Failed to set breakpoint at address {hex(ea)}"
        elif action == 'delete':
            if idaapi.del_bpt(ea):
                return f"Breakpoint deleted at {hex(ea)}"
            return f"Failed to delete breakpoint at address {hex(ea)}"
        elif action == 'enable':
            if enable is None:
                return "Error: enable parameter required for enable action"
            if idaapi.enable_bpt(ea, enable):
                return f"Breakpoint {'enabled' if enable else 'disabled'} at {hex(ea)}"
            return f"Failed to {'' if enable else 'disable '}breakpoint at address {hex(ea)}"
    else:
        return f"Error: Invalid action '{action}'. Supported actions: list, set, delete, enable"

@jsonrpc
@idaread
@unsafe
def dbg_list_breakpoints():
    """列出程序中的所有断点 (已弃用，请使用dbg_manage_breakpoint)"""
    logger.warning("dbg_list_breakpoints is deprecated. Please use dbg_manage_breakpoint with action='list'")
    return dbg_manage_breakpoint('list')

@jsonrpc
@idaread
@unsafe
def dbg_set_breakpoint(
    address: Annotated[str, "在指定地址设置断点"],
) -> str:
    """在指定地址设置断点 (已弃用，请使用dbg_manage_breakpoint)"""
    logger.warning("dbg_set_breakpoint is deprecated. Please use dbg_manage_breakpoint with action='set'")
    return dbg_manage_breakpoint('set', address)

@jsonrpc
@idaread
@unsafe
def dbg_delete_breakpoint(
    address: Annotated[str, "del a breakpoint at the specified address"],
) -> str:
    """del a breakpoint at the specified address (已弃用，请使用dbg_manage_breakpoint)"""
    logger.warning("dbg_delete_breakpoint is deprecated. Please use dbg_manage_breakpoint with action='delete'")
    return dbg_manage_breakpoint('delete', address)

@jsonrpc
@idaread
@unsafe
def dbg_enable_breakpoint(
    address: Annotated[str, "Enable or disable a breakpoint at the specified address"],
    enable: Annotated[bool, "Enable or disable a breakpoint"],
) -> str:
    """Enable or disable a breakpoint at the specified address (已弃用，请使用dbg_manage_breakpoint)"""
    logger.warning("dbg_enable_breakpoint is deprecated. Please use dbg_manage_breakpoint with action='enable'")
    return dbg_manage_breakpoint('enable', address, enable)



def _is_valid_address(address_str: str) -> bool:
    """
    检查地址字符串是否有效
    """
    try:
        parse_address(address_str)
        return True
    except (ValueError, TypeError):
        return False

def _generate_angr_script_template(binary_path: str, func_address_var: str) -> str:
    """
    生成angr脚本基础模板
    """
    return f"""
import angr
import claripy
import sys
import os

# 设置Angr项目
proj = angr.Project('{binary_path}', auto_load_libs=False)

# 函数地址
{func_address_var}

# 创建初始状态
initial_state = proj.factory.entry_state()

# 脚本主体
{{script_body}}

# 运行求解器
{{script_execution}}
"""

def _generate_angr_script_content(script_type: str, func_name: str, func_size: int, options: dict) -> tuple[str, str]:
    """
    根据脚本类型生成相应的脚本内容
    """
    if script_type == 'symbolic_execution':
        arg_count = options.get('arg_count', 1)
        arg_size = options.get('arg_size', 32)
        
        # 生成符号参数
        args_code = []
        call_args = []
        for i in range(arg_count):
            arg_name = f'sym_arg{i+1}'
            args_code.append(f'{arg_name} = claripy.BVS("{arg_name}", {arg_size})  # 参数{i+1}')
            call_args.append(arg_name)
        
        args_code_str = "\n".join(args_code)
        
        script_body = f"""
# 创建函数参数的符号变量
{args_code_str}

# 创建调用函数的状态
state = proj.factory.call_state({func_name}_addr, {', '.join(call_args)})

# 添加约束条件
# Create simulation manager
simgr = proj.factory.simgr(state)
"""
        
        script_execution = """
# 运行符号执行
print("开始符号执行...")
simgr.explore()

# 分析结果
if simgr.deadended:
    print(f"找到 {len(simgr.deadended)} 个终止状态")
    for i, state in enumerate(simgr.deadended):
        print(f"状态 {i}:")
        # 可以根据需要获取返回值或其他信息
        # print(f"  返回值: {state.regs.eax}")
else:
    print("未找到终止状态")
"""
    
    elif script_type == 'brute_force':
        input_size = options.get('input_size', 32)
        success_condition = options.get('success_condition', 'state.regs.eax == 1')
        
        script_body = f"""
# 设置目标函数地址作为 hook 点
def check_password(state):
    # 检查密码是否正确
    if state.solver.is_true({success_condition}):
        print("找到正确密码!")
        # 可以根据需要提取输入值
        # 例如，如果密码存储在内存中的某个位置
        # password_ptr = state.regs.esp + 8  # 假设参数在栈上
        # password = state.memory.load(password_ptr, {input_size})
        # print(f"密码: {state.solver.eval(password, cast_to=bytes)}")
        sys.exit(0)

# 在函数返回前添加 hook
proj.hook({func_name}_addr + {hex(func_size)}, check_password)

# 创建符号输入
sym_input = claripy.BVS('input', 8 * {input_size})  # 输入是{input_size}字节

# 设置符号输入到合适的位置
state = proj.factory.entry_state()
# 例如，设置命令行参数
# state.argv = [proj.filename, sym_input]

# 添加约束以加速求解
# state.solver.add(sym_input.get_byte(0) != 0)  # 非空

# 创建模拟管理器
simgr = proj.factory.simgr(state)
"""
        
        script_execution = """
# 开始爆破
print("开始密码爆破...")
simgr.run()
print("未找到正确密码")
"""
    
    elif script_type == 'control_flow':
        script_body = f"""
# 创建状态
def trace_path(state):
    # 记录路径信息
    path = state.history.bbl_addrs
    print(f"路径长度: {{len(path)}}")
    # 可以根据需要保存路径信息

# 在函数返回前添加 hook
proj.hook({func_name}_addr + {hex(func_size)}, trace_path)

# 创建状态
state = proj.factory.call_state({func_name}_addr)

# 创建模拟管理器
simgr = proj.factory.simgr(state)
"""
        
        script_execution = """
# 运行控制流分析
print("开始控制流分析...")
simgr.run()
"""
    
    else:
        raise IDAError(f"不支持的脚本类型: {{script_type}}")
    
    return script_body, script_execution

@jsonrpc
@idaread
def generate_angr_script(
    function_address: Annotated[str, "要分析的函数地址"],
    script_type: Annotated[str, "脚本类型: 'symbolic_execution', 'brute_force', 'control_flow'"],
    options: Annotated[dict, "可选配置参数，例如输入约束、输出格式等"] = None
) -> str:
    """
    生成angr符号执行或爆破脚本。
    参数：
        function_address: 目标函数地址
        script_type: 脚本类型
        options: 可选配置
    返回：
        生成的Python脚本代码
    """
    if options is None:
        options = {}
    
    # 参数验证
    if not _is_valid_address(function_address):
        raise IDAError(f"无效的函数地址: {{function_address}}")
    
    supported_types = ['symbolic_execution', 'brute_force', 'control_flow']
    if script_type not in supported_types:
        raise IDAError(f"不支持的脚本类型: {{script_type}}。支持的类型: {{', '.join(supported_types)}}")
    
    try:
        # 获取函数信息
        func = idaapi.get_func(parse_address(function_address))
        if not func:
            raise IDAError(f"未找到函数: {{function_address}}")
        
        func_name = idc.get_func_name(func.start_ea)
        if not func_name:
            func_name = f"func_{hex(func.start_ea)}"
        
        # 获取二进制路径
        binary_path = idaapi.get_input_file_path()
        if not binary_path:
            raise IDAError("无法获取二进制文件路径")
        
        # 函数地址变量定义
        func_address_var = f"{func_name}_addr = {{hex(func.start_ea)}}"
        
        # 生成脚本内容
        script_body, script_execution = _generate_angr_script_content(
            script_type, func_name, func.size(), options
        )
        
        # 生成最终脚本
        script = _generate_angr_script_template(binary_path, func_address_var)
        script = script.replace("{{script_body}}", script_body)
        script = script.replace("{{script_execution}}", script_execution)
        
        return script
        
    except Exception as e:
        if isinstance(e, IDAError):
            raise
        logger.error(f"生成angr脚本时出错: {{str(e)}}")
        raise IDAError(f"生成脚本失败: {{str(e)}}")

@jsonrpc
@idaread
def generate_frida_script(
    target: Annotated[str, "目标函数名或地址"],
    script_type: Annotated[str, "脚本类型: 'hook', 'memory_dump', 'string_hook'"],
    options: Annotated[dict, "可选配置参数"] = None
) -> str:
    """
    生成frida动态分析脚本。
    参数：
        target: 目标函数名或地址
        script_type: 脚本类型
        options: 可选配置
    返回：
        生成的JavaScript脚本代码
    """
    # 参数验证
    if options is None:
        options = {}
    
    # 验证脚本类型
    valid_script_types = ['hook', 'memory_dump', 'string_hook']
    if script_type not in valid_script_types:
        raise IDAError(f"不支持的脚本类型: {script_type}。支持的类型: {', '.join(valid_script_types)}")
    
    try:
        # 优先使用script_utils模块生成脚本
        if script_utils is not None:
            print(f"[MCP] 使用script_utils模块生成{script_type}类型的Frida脚本")
            
            # 根据脚本类型调用相应的生成函数
            if script_type == 'hook':
                script_content = script_utils._generate_hook_script(target, options)
            elif script_type == 'memory_dump':
                script_content = script_utils._generate_memory_dump_script(target, options)
            elif script_type == 'string_hook':
                script_content = script_utils._generate_string_hook_script(target, options)
            
            # 获取使用说明
            usage_notes = script_utils._get_usage_notes()
              
            return usage_notes + "\n" + script_content
    except Exception as e:
        # 如果script_utils不可用，使用内置实现
        print("[MCP] script_utils模块不可用，使用内置实现生成Frida脚本")
        
        # 检查target是否为地址
        is_address = False
        try:
            if target.startswith('0x'):
                int(target, 16)
                is_address = True
        except ValueError:
            pass
            
            # 生成脚本模板
            if is_address:
                target_expr = f"ptr('{target}')"
            else:
                target_expr = f"'{target}'"
            
            # 应用类型，默认native（原生二进制）
            app_type = options.get('app_type', 'native')
            
            # 根据应用类型生成不同的前缀和后缀
            if app_type == 'java':
                script_prefix = "Java.perform(function() {\n"
                script_suffix = "\n});"
            else:
                script_prefix = ""  # 原生二进制不需要Java环境
                script_suffix = ""
    
    if script_type == 'hook':
        script_content = f"""
    console.log("开始Hook目标函数: {target}");
    
    // 尝试不同的模块和导出方式
    const moduleName = "{options.get('module', 'target')}";
    const module = Process.getModuleByName(moduleName);
    
    if (!module) {{
        console.log(`未找到模块: ${moduleName}`);
        return;
    }}
    
    let targetFunc = null;
    try {{
        if ({is_address}) {{
            // 如果是地址，直接使用ptr
            targetFunc = {target_expr};
        }} else {{
            // 如果是函数名，尝试在模块中查找导出函数
            targetFunc = Module.findExportByName(moduleName, {target_expr});
            if (!targetFunc) {{
                console.log(`在模块{moduleName}中未找到导出函数: {target}`);
                // 尝试使用模糊搜索查找函数
                console.log("尝试模糊搜索函数...");
                const matches = Memory.scanSync(module.base, module.size, `[${' '.repeat(20)}]${target}${' '.repeat(20)}`);
                if (matches.length > 0) {{
                    console.log(`找到 ${matches.length} 个可能的匹配`);
                    targetFunc = matches[0].address;
                    console.log(`使用第一个匹配: ${targetFunc}`);
                }}
            }}
        }}
        
        if (!targetFunc) {{
            console.log(`未找到目标函数: {target}`);
            return;
        }}
        
        Interceptor.attach(targetFunc, {{
            onEnter: function(args) {{
                console.log(`\n[+] 调用 {target}:`);
                // 打印参数 - 可以根据函数签名调整
                for (let i = 0; i < {options.get('arg_count', 4)}; i++) {{
                    console.log(`  参数 ${i}:`, args[i]);
                    // 尝试将参数解析为字符串
                    try {{
                        const str = Memory.readUtf8String(args[i]);
                        if (str) console.log(`    字符串值: ${str}`);
                    }} catch (e) {{}}
                }}
                // 保存参数供onLeave使用
                this.args = args;
            }},
            onLeave: function(retval) {{
                console.log(`[+] {target} 返回:`);
                console.log(`  返回值:`, retval);
                // 尝试解析返回值为字符串
                try {{
                    const str = Memory.readUtf8String(retval);
                    if (str) console.log(`    字符串值: ${str}`);
                }} catch (e) {{}}
                console.log("----------------------------------------");
            }}
        }});
    }} catch (e) {{
        console.log(`Hook失败: ${e}`);
    }}
"""
        script = script_prefix + script_content + script_suffix
    elif script_type == 'memory_dump':
        script_content = f"""
    console.log("开始内存监控...");
    
    const targetAddr = {target_expr};
    const memSize = {options.get('size', 1024)}; // 要监控的内存大小
    
    console.log(`监控地址范围: ${targetAddr} - ${ptr(targetAddr).add(memSize)}`);
    
    try {{
        // 监控内存读写
        Memory.protect(ptr(targetAddr), memSize, 'rwx');
        
        // Windows下可能需要特殊处理
        if (Process.platform === 'windows') {{
            console.log("Windows平台: 使用内存断点方式监控");
            // 为内存区域设置读写断点
            Memory.watchpoint(ptr(targetAddr), memSize, {{ 
                read: true,
                write: true,
                onAccess: function(details) {{
                    console.log(`\n[+] 内存访问:`);
                    console.log(`  地址: ${details.address}`);
                    console.log(`  类型: ${details.type}`); // 'read' 或 'write'
                    console.log(`  大小: ${details.size} 字节`);
                    
                    // 打印调用栈
                    const stack = Thread.backtrace(this.context, Backtracer.ACCURATE)
                        .map(DebugSymbol.fromAddress).join('\n');
                    console.log(`  调用栈:\n${stack}`);
                    
                    // 对于写操作，尝试打印写入的值
                    if (details.type === 'write' && details.data) {{
                        console.log(`  写入值:`, details.data);
                    }}
                    console.log("----------------------------------------");
                }}
            }});
        }} else {{
            // 其他平台使用accessMonitor
            Memory.accessMonitor.enable();
            
            // 监听内存访问事件
            Memory.accessMonitor.on('access', function(event) {{
                if (event.address.compare(ptr(targetAddr)) >= 0 && 
                    event.address.compare(ptr(targetAddr).add(memSize)) < 0) {{
                    console.log(`\n[+] 内存访问:`);
                    console.log(`  地址: ${event.address}`);
                    console.log(`  类型: ${event.type}`); // 'read' 或 'write'
                    console.log(`  大小: ${event.size} 字节`);
                    
                    // 打印调用栈
                    const stack = Thread.backtrace(event.thread, Backtracer.ACCURATE)
                        .map(DebugSymbol.fromAddress).join('\n');
                    console.log(`  调用栈:\n${stack}`);
                    
                    // 对于写操作，尝试打印写入的值
                    if (event.type === 'write') {{
                        try {{
                            console.log(`  写入值:`, Memory.readByteArray(event.address, event.size));
                        }} catch (e) {{}}
                    }}
                    console.log("----------------------------------------");
                }}
            }});
        }}
        
        console.log("内存监控已启动。按Ctrl+C停止。");
    }} catch (e) {{
        console.log(`内存监控设置失败: ${e}`);
    }}
"""
        script = script_prefix + script_content + script_suffix
    elif script_type == 'string_hook':
        script_content = f"""
    console.log("开始字符串监控...");
    
    // 存储已收集的字符串
    const collectedStrings = new Set();
    
    // Hook常见的字符串处理函数
    const stringFuncs = {{
        'strcmp': Module.findExportByName(null, 'strcmp'),
        'strcpy': Module.findExportByName(null, 'strcpy'),
        'strlen': Module.findExportByName(null, 'strlen'),
        'memcpy': Module.findExportByName(null, 'memcpy'),
        'strcat': Module.findExportByName(null, 'strcat')
    }};
    
    // Hook目标函数附近的字符串操作
    if ({is_address}) {{
        // 如果指定了地址，也监控该地址附近的内存读取
        try {{
            const targetAddr = {target_expr};
            console.log(`监控地址附近的字符串: ${targetAddr}`);
            Memory.scan(ptr(targetAddr).sub(0x1000), 0x2000, "[41-7a]{4,}", {{
                onMatch: function(address, size) {{
                    try {{
                        const str = Memory.readUtf8String(address);
                        if (str.length >= 4 && !collectedStrings.has(str)) {{
                            collectedStrings.add(str);
                            console.log(`\n[+] 发现字符串:`);
                            console.log(`  地址: ${address}`);
                            console.log(`  内容: "${str}"`);
                            
                            // 获取调用栈
                            const stack = Thread.backtrace(Thread.currentThread(), Backtracer.ACCURATE)
                                .map(DebugSymbol.fromAddress).join('\n');
                            console.log(`  访问栈:\n${stack}`);
                        }}
                    }} catch (e) {{}}
                }},
                onComplete: function() {{}}
            }});
        }} catch (e) {{
            console.log(`字符串扫描失败: ${e}`);
        }}
    }}
    
    // Hook字符串函数
    for (const [name, func] of Object.entries(stringFuncs)) {{
        if (func) {{
            Interceptor.attach(func, {{
                onEnter: function(args) {{
                    try {{
                        // 尝试读取第一个参数作为字符串
                        const str = Memory.readUtf8String(args[0]);
                        if (str && str.length >= 3 && !collectedStrings.has(str)) {{
                            collectedStrings.add(str);
                            console.log(`\n[+] 函数 {name} 使用字符串:`);
                            console.log(`  字符串: "${str}"`);
                            
                            // 获取调用栈
                            const stack = Thread.backtrace(this.context, Backtracer.ACCURATE)
                                .map(DebugSymbol.fromAddress).join('\n');
                            console.log(`  调用栈:\n${stack}`);
                        }}
                    }} catch (e) {{}}
                }}
            }});
        }}
    }}
    
    console.log("字符串监控已启动。按Ctrl+C停止。");
    console.log("已收集的字符串将自动去重并显示。");
"""
        script = script_prefix + script_content + script_suffix
    else:
        raise IDAError(f"不支持的脚本类型: {script_type}")
    
    usage_notes = """// 使用说明:
// 1. 确保已安装frida-tools: pip install frida-tools
// 2. 对于Windows原生程序，使用以下命令运行:
//    frida -p <进程ID> -l <脚本文件> --no-pause
//    或附加到已运行的程序
// 3. 对于Java程序，使用:
//    frida -U -f <包名> -l <脚本文件> --no-pause
// 4. 若要保存输出，可重定向到文件:
//    frida -p <进程ID> -l <脚本文件> --no-pause > output.log
"""
    
    return usage_notes + "\n" + script

@jsonrpc
@idawrite
def save_generated_script(
    script_content: Annotated[str, "脚本内容"],
    script_type: Annotated[str, "脚本类型: 'angr' 或 'frida'"],
    file_name: Annotated[str, "保存的文件名（可选）"] = None
) -> str:
    """
    保存生成的脚本到文件。
    参数：
        script_content: 脚本内容
        script_type: 脚本类型
        file_name: 保存的文件名（可选）
    返回：
        保存的文件路径
    """
    import os
    
    # 确定文件扩展名
    if script_type == 'angr':
        extension = '.py'
    elif script_type == 'frida':
        extension = '.js'
    else:
        raise IDAError(f"不支持的脚本类型: {script_type}")
    
    # 确定保存目录（使用临时目录或IDA当前目录）
    save_dir = os.path.dirname(idaapi.get_input_file_path()) if idaapi.get_input_file_path() else os.getcwd()
    
    # 确定文件名
    if not file_name:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{script_type}_script_{timestamp}{extension}"
    elif not file_name.endswith(extension):
        file_name += extension
    
    # 构建完整文件路径
    file_path = os.path.join(save_dir, file_name)
    
    # 保存文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        return f"脚本已保存至: {file_path}\n\n使用以下命令运行:\n{get_run_command_hint(script_type, file_path)}"
    except Exception as e:
        raise IDAError(f"保存脚本失败: {str(e)}")

@jsonrpc
@idawrite
def run_external_script(
    script_path: Annotated[str, "脚本文件路径"],
    script_type: Annotated[str, "脚本类型: 'angr' 或 'frida'"],
    target_binary: Annotated[str, "目标二进制文件路径"] = None,
    target_pid: Annotated[int, "目标进程ID（可选，用于frida）"] = None
) -> str:
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
    import subprocess
    import tempfile
    import os
    
    # 检查脚本文件是否存在
    if not os.path.exists(script_path):
        raise IDAError(f"脚本文件不存在: {script_path}")
    
    # 如果未提供目标二进制，使用当前加载的二进制
    if target_binary is None:
        target_binary = idaapi.get_input_file_path()
        if not target_binary:
            raise IDAError("未加载二进制文件，无法确定目标")
    
    # 构建命令
    if script_type == 'angr':
        # 确保angr模块可用
        try:
            import angr
        except ImportError:
            raise IDAError("未安装angr模块，请先安装: pip install angr")
        
        # 使用当前Python环境运行脚本
        cmd = [sys.executable, script_path]
    elif script_type == 'frida':
        # 确保frida模块可用
        try:
            # 检查frida命令是否在PATH中
            subprocess.run(['frida', '--version'], check=True, capture_output=True, text=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise IDAError("未安装frida命令行工具，请先安装: pip install frida-tools")
        
        # 构建frida命令
        if target_pid is not None:
            # 附加到已运行的进程
            cmd = ['frida', '-p', str(target_pid), '-l', script_path, '--no-pause']
        else:
            # 在Windows上，使用spawn模式可能需要管理员权限
            cmd = ['frida', '-f', target_binary, '-l', script_path, '--no-pause']
            # 添加Windows特有的提示
            if sys.platform == 'win32':
                print("注意：在Windows上以spawn模式运行frida可能需要管理员权限")
                print("建议先手动启动目标程序，然后使用-p参数附加到进程")
    else:
        raise IDAError(f"不支持的脚本类型: {script_type}")
    
    try:
        # 创建临时文件保存输出
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log', encoding='utf-8') as temp_log:
            log_path = temp_log.name
        
        # 运行脚本并捕获输出
        print(f"运行{script_type}脚本: {script_path} 目标: {target_binary} {f'PID: {target_pid}' if target_pid else ''}")
        
        # 对于frida，考虑到它可能需要用户交互，我们不使用capture_output，而是直接输出
        if script_type == 'frida':
            output = f"正在运行frida脚本，请在终端中查看输出...\n"
            output += f"命令: {' '.join(cmd)}\n\n"
            output += "注意：\n"
            output += "1. frida脚本在Windows上可能需要管理员权限\n"
            output += "2. 建议使用以下方式手动运行，以获得更好的交互体验:\n"
            output += f"   {' '.join(cmd)}\n\n"
            output += "3. 脚本执行结果将不会在此处显示，请在独立终端中运行以查看完整输出"
            
            # 在后台启动frida进程，但不等待其完成
            try:
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )
            except Exception as e:
                output += f"\n启动进程失败: {str(e)}"
            
            return output
        else:
            # 对于angr，我们可以捕获输出
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 增加超时时间到5分钟，因为angr分析可能需要较长时间
            )
            
            # 保存输出到日志文件
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"命令: {' '.join(cmd)}\n\n")
                f.write(f"返回码: {result.returncode}\n\n")
                f.write("标准输出:\n")
                f.write(result.stdout)
                f.write("\n\n标准错误:\n")
                f.write(result.stderr)
            
            # 返回执行结果摘要
            output = f"脚本执行 {'成功' if result.returncode == 0 else '失败'}。\n"
            output += f"返回码: {result.returncode}\n"
            output += f"输出日志已保存至: {log_path}\n\n"
            
            # 显示部分输出
            if result.stdout:
                output += "标准输出 (前5行):\n"
                output += '\n'.join(result.stdout.split('\n')[:5]) + '\n...\n\n'
            
            if result.stderr:
                output += "标准错误:\n"
                output += result.stderr
            
            return output
    except subprocess.TimeoutExpired:
        return f"脚本执行超时（>5分钟）。\n请手动运行脚本以查看完整输出。"
    except Exception as e:
        raise IDAError(f"运行脚本时发生错误: {str(e)}")

@idaread
def get_run_command_hint(script_type: str, script_path: str) -> str:
    """
    获取运行脚本的命令提示。
    """
    if script_type == 'angr':
        return f"python {script_path}"
    elif script_type == 'frida':
        # 获取当前加载的二进制文件路径
        binary_path = idaapi.get_input_file_path()
        if binary_path:
            return f"frida -f {binary_path} -l {script_path} --no-pause  # 启动新进程\n或\nfrida -p <进程ID> -l {script_path} --no-pause  # 附加到已运行进程"
        else:
            return f"frida -f <二进制文件路径> -l {script_path} --no-pause  # 启动新进程\n或\nfrida -p <进程ID> -l {script_path} --no-pause  # 附加到已运行进程"

class MCP(idaapi.plugin_t):
    """MCP插件主类，增强版实现"""
    flags = idaapi.PLUGIN_KEEP
    comment = "MCP Plugin"
    help = "MCP"
    wanted_name = "MCP"
    wanted_hotkey = "Ctrl-Alt-M"

    def init(self):
        """初始化插件，创建服务器实例"""
        try:
            # 创建服务器实例，增加错误处理
            try:
                self.server = Server()
                self.is_initialized = True
            except Exception as e:
                print(f"[MCP] 警告: 创建服务器实例时出错: {str(e)}")
                print("[MCP] 尝试使用备用初始化...")
                # 创建一个最小的服务器对象作为后备
                class FallbackServer:
                    def __init__(self):
                        self.running = False
                    def start(self):
                        print("[MCP] 警告: 使用备用服务器实现")
                        self.running = True
                    def stop(self):
                        self.running = False
                self.server = FallbackServer()
                self.is_initialized = True
            
            # 格式化热键显示
            hotkey = MCP.wanted_hotkey.replace("-", "+")
            if sys.platform == "darwin":
                hotkey = hotkey.replace("Alt", "Option")
            elif sys.platform == "win32":
                hotkey = hotkey.replace("Ctrl", "Ctrl")
            
            # 显示插件加载信息
            print(f"[MCP] Plugin loaded, use Edit -> Plugins -> MCP ({hotkey}) to start the server")
            print("[MCP] 版本增强: 添加了自动错误恢复机制和详细日志记录")
            
            return idaapi.PLUGIN_KEEP
            
        except Exception as e:
            # 捕获所有初始化异常
            import traceback
            print(f"[MCP] 错误: 插件初始化失败: {str(e)}")
            print("[MCP] 详细错误:")
            traceback.print_exc()
            # 即使初始化失败，也尽量保持插件可用
            return idaapi.PLUGIN_KEEP

    def run(self, args):
        """运行插件，启动服务器，增加错误处理和状态检查"""
        try:
            if not hasattr(self, 'server') or self.server is None:
                print("[MCP] 错误: 服务器实例不存在")
                return
            
            print("[MCP] 正在启动服务器...")
            
            # 尝试启动服务器
            try:
                self.server.start()
                
                # 短暂延迟以确保服务器有时间初始化
                import time
                time.sleep(0.3)
                
                # 检查服务器状态
                if hasattr(self.server, 'running') and self.server.running:
                    status = "已启动成功"
                    # 显示端口信息
                    if hasattr(self.server, 'port'):
                        status += f"，监听端口: {self.server.port}"
                else:
                    status = "启动状态未知，请检查日志"
                
                print(f"[MCP] 服务器{status}")
                
            except Exception as e:
                print(f"[MCP] 错误: 启动服务器时出错: {str(e)}")
                print("[MCP] 建议检查端口占用情况或重启IDA Pro")
                
        except Exception as e:
            print(f"[MCP] 错误: 执行run方法时发生未知错误: {str(e)}")

    def term(self):
        """终止插件，停止服务器，增加优雅关闭和错误恢复"""
        try:
            if hasattr(self, 'server') and self.server is not None:
                print("[MCP] 正在停止服务器...")
                try:
                    self.server.stop()
                    print("[MCP] 服务器已停止")
                except Exception as e:
                    print(f"[MCP] 警告: 停止服务器时出错: {str(e)}")
        except Exception as e:
            # 即使停止出错也不要影响IDA的关闭
            print(f"[MCP] 警告: 终止插件时出错: {str(e)}")


def PLUGIN_ENTRY():
    """插件入口点，增加错误处理"""
    try:
        plugin = MCP()
        return plugin
    except Exception as e:
        print(f"[MCP] 错误: 创建插件实例失败: {str(e)}")
        # 返回一个最小的插件实例作为后备
        class MinimalPlugin(idaapi.plugin_t):
            flags = idaapi.PLUGIN_KEEP
            comment = "MCP (Minimal)"
            help = "MCP Minimal Version"
            wanted_name = "MCP"
            wanted_hotkey = ""
            
            def init(self):
                print("[MCP] 已加载最小化版本，请检查完整版本的安装")
                return idaapi.PLUGIN_KEEP
            
            def run(self, args):
                print("[MCP] 最小化版本不提供完整功能，请重新安装插件")
            
            def term(self):
                pass
        
        return MinimalPlugin()

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
