# Automata-mcp-server

一个基于 FastAPI 和 Model Context Protocol (MCP) 的插件化服务器。

## 功能特性

- 🔌 插件化架构，支持动态加载工具
- 🔄 **热重载功能**，开发时自动监听代码变化并重启服务器
- 🚀 基于 FastAPI，高性能异步处理
- 🔐 支持 API Key 认证
- 📦 支持自动安装工具依赖

## 快速开始

### 1. 安装依赖

使用 uv（推荐）
```bash
uv sync
```

**📝 依赖管理说明**：
- 所有依赖在 `pyproject.toml` 中管理
- `requirements.txt` 由 `pyproject.toml` 自动生成
- Git pre-commit 钩子会自动同步两个文件
- 详见：[scripts/README.md](scripts/README.md)

### 2. 配置环境变量

复制示例配置文件并修改：
```bash
cp .env.example .env
```

编辑 `.env` 文件，设置必要的配置项。

### 3. 启动服务器

#### 开发模式（启用热重载）

使用 uv：
```bash
uv run main.py
```

或直接运行：
```bash
python main.py
```

服务器将在开发模式下启动，自动监听项目中所有 Python 文件的变化。当检测到 `.py` 文件修改时，服务器会自动重启，无需手动重启。

#### 生产模式（禁用热重载）

如需在生产环境中运行，可以直接使用 uvicorn：
```bash
uvicorn app.server:app --host $HOST --port $PORT
```

## 开发说明

### 热重载功能

服务器默认启用热重载功能，会监听项目中所有 Python 文件（`**/*.py`）的变化。当任何 `.py` 文件发生变化时，服务器会自动重启并加载最新代码。

**详细的开发指南请查看：[docs/development.md](docs/development.md)**

### 添加新工具

1. 在 `src/` 或 `app/AutoUp-MCP-Extension/` 目录下创建新的工具包
2. 继承 `BaseMCPTool` 类并实现工具逻辑
3. 添加 `config.yaml` 配置文件
4. 服务器会自动发现并加载新工具（开发模式下会自动重载）

## API 文档

启动服务器后，访问以下地址查看自动生成的 API 文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 项目结构

```
.
├── app/                          # 核心应用代码
│   ├── server.py                 # 服务器主文件
│   ├── base_tool.py             # 工具基类
│   ├── routers.py               # 路由定义
│   └── AutoUp-MCP-Extension/    # 扩展工具目录
├── src/                          # 用户工具目录
│   └── fetch/                   # 示例工具
├── main.py                      # 入口文件
├── pyproject.toml              # 项目配置
├── .env.example                # 环境变量示例
└── README.md                   # 项目说明
```

## 许可证

MIT
