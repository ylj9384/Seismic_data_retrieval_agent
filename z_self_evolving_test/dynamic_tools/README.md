# dynamic_tools 目录说明
此目录用于存放“模型自动生成并通过安全校验”的工具源码文件，命名格式：
  tool_<name>.py

文件结构规范：
- 仅包含一个顶层函数定义（与注册名一致）。
- 函数对象在生成时被打标记：_is_dynamic_tool = True
- 不允许出现类定义 / with / try / exec / eval / open 等敏感结构。
- 代码只可导入白名单模块：obspy / numpy / math / typing / collections（若生成阶段允许的范围后续扩展再调整）。

运行流程：
1. 模型输出 propose_tool JSON。
2. parser.extract_function 安全解析并写入本目录。
3. registry._load_dynamic_tools 在启动时重新导入这些函数并注册。

安全提示：
- 不要手动放入未经审计的脚本；被加载后将被执行。
- 若需要下架某工具，删除对应文件并在 tools_meta.json 中去除其元数据或让系统重新保存。

版本控制建议：
- 可将本目录纳入版本管理以追踪模型自动生成的演化过程。