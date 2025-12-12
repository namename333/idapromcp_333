# IDA Pro MCP 智能逆向分析平台（增强版）
# 基于 mrexodia/ida-pro-mcp 二次开发增强

__version__ = "1.4.0"
__author__ = "namename333"
__description__ = "IDA Pro MCP 智能逆向分析平台，支持多客户端和自动化分析"

# 导入主要模块，方便直接使用
from .server import main as server_main
try:
    from .idalib_server import main as idalib_main
except ImportError:
    idalib_main = None

# 导出主要功能
__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "server_main",
    "idalib_main",
]