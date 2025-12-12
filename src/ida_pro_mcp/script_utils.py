"""
脚本生成工具模块
提供各种辅助函数，用于生成不同类型的脚本代码
"""

def _is_address_string(target: str) -> bool:
    """
    检查目标字符串是否表示一个有效的十六进制地址
    
    参数:
        target: 要检查的字符串
    返回:
        bool: 如果是有效的十六进制地址则返回True
    """
    try:
        if target.startswith('0x'):
            int(target, 16)
            return True
    except ValueError:
        pass
    return False

def _get_target_expression(target: str, is_address: bool) -> str:
    """
    根据目标类型生成Frida表达式
    
    参数:
        target: 目标函数名或地址
        is_address: 是否为地址
    返回:
        str: 适合在Frida脚本中使用的表达式
    """
    if is_address:
        return f"ptr('{target}')"
    return f"'{target}'"

def _get_app_environment(app_type: str) -> tuple[str, str]:
    """
    根据应用类型获取环境前缀和后缀
    
    参数:
        app_type: 应用类型 ('native' 或 'java')
    返回:
        tuple: (script_prefix, script_suffix)
    """
    if app_type == 'java':
        return "Java.perform(function() {\n", "\n});"
    return "", ""

def _generate_hook_script(target: str, is_address: bool, options: dict) -> str:
    """
    生成函数Hook脚本
    
    参数:
        target: 目标函数名或地址
        is_address: 是否为地址
        options: 配置选项
    返回:
        str: Hook脚本内容
    """
    # 根据是否为地址生成目标表达式
    target_expr = _get_target_expression(target, is_address)
    module_name = options.get('module', 'target')
    arg_count = options.get('arg_count', 4)
    
    return f"""
    console.log("开始Hook目标函数: {target}");
    
    // 尝试不同的模块和导出方式
    const moduleName = "{module_name}";
    const module = Process.getModuleByName(moduleName);
    
    if (!module) {{
        console.log(`未找到模块: ${{moduleName}}`);
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
                console.log(`在模块${{moduleName}}中未找到导出函数: {target}`);
                // 尝试使用模糊搜索查找函数
                console.log("尝试模糊搜索函数...");
                const matches = Memory.scanSync(module.base, module.size, `[${' ' * 20}]{target}${' ' * 20}`);
                if (matches.length > 0) {{
                    console.log(`找到 ${{matches.length}} 个可能的匹配`);
                    targetFunc = matches[0].address;
                    console.log(`使用第一个匹配: ${{targetFunc}}`);
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
                for (let i = 0; i < {arg_count}; i++) {{
                    console.log(`  参数 ${{i}}:`, args[i]);
                    // 尝试将参数解析为字符串
                    try {{
                        const str = Memory.readUtf8String(args[i]);
                        if (str) console.log(`    字符串值: ${{str}}`);
                    }} catch (e) {{
                        // 静默失败，不影响主流程
                    }}
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
                    if (str) console.log(`    字符串值: ${{str}}`);
                }} catch (e) {{
                    // 静默失败，不影响主流程
                }}
                console.log("----------------------------------------");
            }}
        }});
    }} catch (e) {{
        console.log(`Hook失败: ${{e}}`);
    }}
"""

def _generate_memory_dump_script(target: str, options: dict) -> str:
    """
    生成内存监控脚本
    
    参数:
        target: 目标地址
        options: 配置选项
    返回:
        str: 内存监控脚本内容
    """
    mem_size = options.get('size', 1024)
    
    # 判断是否为地址字符串
    is_address = _is_address_string(target)
    target_expr = _get_target_expression(target, is_address)
    
    return f"""
    console.log("开始内存监控...");
    
    const targetAddr = {target_expr};
    const memSize = {mem_size}; // 要监控的内存大小
    
    console.log(`监控地址范围: ${{targetAddr}} - ${{ptr(targetAddr).add(memSize)}}`);
    
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
                    console.log(`  地址: ${{details.address}}`);
                    console.log(`  类型: ${{details.type}}`); // 'read' 或 'write'
                    console.log(`  大小: ${{details.size}} 字节`);
                    
                    // 打印调用栈
                    try {{
                        const stackTrace = Thread.backtrace(this.context, Backtracer.ACCURATE)
                            .map(DebugSymbol.fromAddress).join('\n');
                        console.log(`  调用栈:\n${{stackTrace}}`);
                    }} catch (e) {{
                        console.log(`  调用栈获取失败: ${{e}}`);
                    }}
                    
                    // 对于写操作，尝试打印写入的值
                    if (details.type === 'write') {{
                        try {{
                            console.log(`  写入值:`, Memory.readByteArray(details.address, details.size));
                        }} catch (e) {{
                            // 静默失败，不影响主流程
                        }}
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
                    console.log(`  地址: ${{event.address}}`);
                    console.log(`  类型: ${{event.type}}`); // 'read' 或 'write'
                    console.log(`  大小: ${{event.size}} 字节`);
                    
                    // 打印调用栈
                    try {{
                        const stackTrace = Thread.backtrace(event.thread, Backtracer.ACCURATE)
                            .map(DebugSymbol.fromAddress).join('\n');
                        console.log(`  调用栈:\n${{stackTrace}}`);
                    }} catch (e) {{
                        console.log(`  调用栈获取失败: ${{e}}`);
                    }}
                    
                    // 对于写操作，尝试打印写入的值
                    if (event.type === 'write') {{
                        try {{
                            console.log(`  写入值:`, Memory.readByteArray(event.address, event.size));
                        }} catch (e) {{
                            // 静默失败，不影响主流程
                        }}
                    }}
                    console.log("----------------------------------------");
                }}
            }});
        }}
        
        console.log("内存监控已启动。按Ctrl+C停止。");
    }} catch (e) {{
        console.log(`内存监控设置失败: ${{e}}`);
    }}
"""

def _generate_string_hook_script(target: str, is_address: bool, options: dict = None) -> str:
    """
    生成字符串监控脚本
    
    参数:
        target: 目标函数名或地址
        is_address: 是否为地址
        options: 配置选项 (可选)
    返回:
        str: 字符串监控脚本内容
    """
    # 根据是否为地址生成目标表达式
    target_expr = _get_target_expression(target, is_address)
    # 如果options为None，初始化为空字典
    if options is None:
        options = {}
    memory_scan_code = """
    // Hook目标函数附近的字符串操作
    if ({is_address}) {{
        // 如果指定了地址，也监控该地址附近的内存读取
        try {{
            const targetAddr = {target_expr};
            console.log(`监控地址附近的字符串: ${{targetAddr}}`);
            Memory.scan(ptr(targetAddr).sub(0x1000), 0x2000, "[41-7a]{{4,}}", {{
                onMatch: function(address, size) {{
                    try {{
                        const str = Memory.readUtf8String(address);
                        if (str.length >= 4 && !collectedStrings.has(str)) {{
                            collectedStrings.add(str);
                            console.log(`\n[+] 发现字符串:`);
                            console.log(`  地址: ${{address}}`);
                            console.log(`  内容: "${{str}}"`);
                            
                            // 获取调用栈
                            try {{
                                const stackTrace = Thread.backtrace(Thread.currentThread(), Backtracer.ACCURATE)
                                    .map(DebugSymbol.fromAddress).join('\n');
                                console.log(`  访问栈:\n${{stackTrace}}`);
                            }} catch (e) {{
                                console.log(`  调用栈获取失败: ${{e}}`);
                            }}
                        }}
                    }} catch (e) {{
                        // 静默失败，继续扫描
                    }}
                }},
                onComplete: function() {{}}
            }});
        }} catch (e) {{
            console.log(`字符串扫描失败: ${{e}}`);
        }}
    }}
"""
    if not is_address:
        memory_scan_code = ""
        
    return f"""
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
    
    {memory_scan_code}
    
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
                            console.log(`\n[+] 函数 {{name}} 使用字符串:`);
                            console.log(`  字符串: "${{str}}"`);
                            
                            // 获取调用栈
                            try {{
                                const stackTrace = Thread.backtrace(this.context, Backtracer.ACCURATE)
                                    .map(DebugSymbol.fromAddress).join('\n');
                                console.log(`  调用栈:\n${{stackTrace}}`);
                            }} catch (e) {{
                                console.log(`  调用栈获取失败: ${{e}}`);
                            }}
                        }}
                    }} catch (e) {{
                        // 静默失败，继续处理
                    }}
                }}
            }});
        }}
    }}
    
    console.log("字符串监控已启动。按Ctrl+C停止。");
    console.log("已收集的字符串将自动去重并显示。");
"""

def _get_usage_notes() -> str:
    """
    获取使用说明注释
    
    返回:
        str: 使用说明文本
    """
    return """// 使用说明:
// 1. 确保已安装frida-tools: pip install frida-tools
// 2. 对于Windows原生程序，使用以下命令运行:
//    frida -p <进程ID> -l <脚本文件> --no-pause
//    或附加到已运行的程序
// 3. 对于Java程序，使用:
//    frida -U -f <包名> -l <脚本文件> --no-pause"""
