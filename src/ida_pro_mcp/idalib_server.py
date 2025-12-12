import sys
import inspect
import logging
import argparse
import importlib
import os
import json
from pathlib import Path
import typing_inspection.introspection as intro

from mcp.server.fastmcp import FastMCP

# idapro must go first to initialize idalib
import idapro

import ida_auto
import ida_hexrays

logger = logging.getLogger(__name__)

mcp = FastMCP("github.com/namename333/idapromcp_333#idalib")

def get_config_file_path():
    """
    获取配置文件路径
    支持多种配置文件位置
    """
    # 优先检查用户目录
    user_dir = os.path.expanduser("~")
    config_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_config.json"),
        os.path.join(user_dir, ".mcp_config.json"),
        os.path.join(user_dir, "mcp_config.json")
    ]
    
    # 检查是否存在IDA插件目录下的配置文件
    try:
        import ida_idaapi
        plugin_dir = ida_idaapi.idadir("plugins")
        config_paths.append(os.path.join(plugin_dir, "mcp_config.json"))
    except ImportError:
        pass  # 如果不在IDA环境中运行，忽略
    
    # 返回第一个存在的配置文件
    for path in config_paths:
        if os.path.exists(path):
            return path
    
    # 如果没有找到配置文件，返回默认路径
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_config.json")

def load_config():
    """
    加载配置文件
    返回配置字典，如果配置文件不存在则返回默认配置
    """
    default_config = {
        "host": "127.0.0.1",
        "port": 8745  # idalib服务器默认端口为8745
    }
    
    config_path = get_config_file_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 合并默认配置和用户配置
                default_config.update(user_config)
        except Exception as e:
            print(f"警告: 加载配置文件失败: {e}")
    
    # 从环境变量覆盖配置（使用IDALIB_MCP_PORT区分）
    if "MCP_HOST" in os.environ:
        default_config["host"] = os.environ["MCP_HOST"]
    
    if "IDALIB_MCP_PORT" in os.environ:
        try:
            default_config["port"] = int(os.environ["IDALIB_MCP_PORT"])
        except ValueError:
            print("警告: 环境变量IDALIB_MCP_PORT不是有效的端口号")
    elif "MCP_PORT" in os.environ:
        # 如果没有指定IDALIB_MCP_PORT，也可以使用通用的MCP_PORT
        try:
            default_config["port"] = int(os.environ["MCP_PORT"])
        except ValueError:
            print("警告: 环境变量MCP_PORT不是有效的端口号")
    
    return default_config

def fixup_tool_argument_descriptions(mcp: FastMCP):
    # 在`mcp-plugin.py`的工具定义中，我们在函数参数上使用`typing.Annotated`
# 来附加文档。例如：
    #
    #     def get_function_by_name(
    #         name: Annotated[str, "Name of the function to get"]
    #     ) -> Function:
    #         """Get a function by its name"""
    #         ...
    #
    # 然而，Annotated的解释权由静态分析工具和其他工具决定。
    # FastMCP对这些注释没有特殊处理，因此我们需要手动将它们添加到工具元数据中。
    #
    # 示例：修改前
    #
    #     tool.parameter={
    #       properties: {
    #         name: {
    #           title: "Name",
    #           type: "string"
    #         }
    #       },
    #       required: ["name"],
    #       title: "get_function_by_nameArguments",
    #       type: "object"
    #     }
    #
    # 示例：修改后
    #
    #     tool.parameter={
    #       properties: {
    #         name: {
    #           title: "Name",
    #           type: "string"
    #           description: "Name of the function to get"
    #         }
    #       },
    #       required: ["name"],
    #       title: "get_function_by_nameArguments",
    #       type: "object"
    #     }
    #
    # 参考资料：
    #   - https://docs.python.org/3/library/typing.html#typing.Annotated
    #   - https://fastapi.tiangolo.com/python-types/#type-hints-with-metadata-annotations

    # 遗憾的是，FastMCP.list_tools()是异步的，因此我们违背最佳实践直接访问`._tool_manager`
    # 而不是为了获取（非异步的）工具列表而启动asyncio运行时。
    for tool in mcp._tool_manager.list_tools():
        sig = inspect.signature(tool.fn)
        for name, parameter in sig.parameters.items():
            # this instance is a raw `typing._AnnotatedAlias` that we can't do anything with directly.
            # it renders like:
            #
            #      typing.Annotated[str, 'Name of the function to get']
            if not parameter.annotation:
                continue

            # this instance will look something like:
            #
            #     InspectedAnnotation(type=<class 'str'>, qualifiers=set(), metadata=['Name of the function to get'])
            #
            annotation = intro.inspect_annotation(
                                                  parameter.annotation,
                                                  annotation_source=intro.AnnotationSource.ANY
                                              )

            # for our use case, where we attach a single string annotation that is meant as documentation,
            # we extract that string and assign it to "description" in the tool metadata.

            if annotation.type is not str:
                continue

            if len(annotation.metadata) != 1:
                continue

            description = annotation.metadata[0]
            if not isinstance(description, str):
                continue

            logger.debug("adding parameter documentation %s(%s='%s')", tool.name, name, description)
            tool.parameters["properties"][name]["description"] = description

def main():
    # 先加载配置文件，作为命令行参数的默认值
    config = load_config()
    
    parser = argparse.ArgumentParser(description="MCP server for IDA Pro via idalib")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show debug messages")
    parser.add_argument("--host", type=str, default=config["host"], help=f"Host to listen on, default: {config['host']}")
    parser.add_argument("--port", type=int, default=config["port"], help=f"Port to listen on, default: {config['port']}")
    parser.add_argument("input_path", type=Path, help="Path to the input file to analyze.")
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG 
        idapro.enable_console_messages(True)
    else:
        log_level = logging.INFO
        idapro.enable_console_messages(False)

    mcp.settings.log_level = logging.getLevelName(log_level)
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    logging.basicConfig(level=log_level)

    # 重置可能在idapythonrc.py中初始化的日志级别
    # 该文件在导入idalib时会被执行
    logging.getLogger().setLevel(log_level)

    if not args.input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input_path}")

    # TODO: add a tool for specifying the idb/input file (sandboxed)
    logger.info("正在打开数据库: %s", args.input_path)
    if idapro.open_database(str(args.input_path), run_auto_analysis=True):
        raise RuntimeError("分析输入文件失败")

    logger.debug("idalib: waiting for analysis...")
    ida_auto.auto_wait()

    if not ida_hexrays.init_hexrays_plugin():
        raise RuntimeError("Hex-Rays反编译器初始化失败")

    plugin = importlib.import_module("ida_pro_mcp.mcp-plugin")
    logger.debug("正在添加工具...")
    # 无条件添加所有功能（包括unsafe）
    for name, callable in plugin.rpc_registry.methods.items():
        logger.debug("添加工具: %s: %s", name, callable)
        mcp.add_tool(callable, name)

    # NOTE: https://github.com/modelcontextprotocol/python-sdk/issues/466
    fixup_tool_argument_descriptions(mcp)

    # NOTE: npx @modelcontextprotocol/inspector for debugging
    logger.info("MCP服务器在: http://%s:%d/sse 可用", mcp.settings.host, mcp.settings.port)
    try:
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
