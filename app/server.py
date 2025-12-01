import importlib
import inspect
import os
import subprocess
from pathlib import Path

# Core server module for Automata MCP Server
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_mcp import FastApiMCP
from loguru import logger
from pydantic import BaseModel

from .base_tool import BaseMCPTool
from .exceptions import (
    AutomataError,
    ConfigurationError,
    DependencyInstallError,
    ToolLoadError,
    handle_exception,
    with_exception_handling,
)
from .routers import (
    create_router,
    create_tool_endpoint,
    get_route_configs,
    verify_api_key_dependency,
)


class MCPRequest(BaseModel):
    method: str
    params: dict | None = None
    id: str | None = None


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: dict | None = None
    error: dict | None = None
    id: str | None = None


class AutomataMCPServer:
    def __init__(self):
        # Derive the OpenAPI `servers` entry from environment variables to avoid
        # hard-coded addresses. Prefer an explicit SERVER_URL if provided,
        # otherwise build from HOST and PORT with sensible defaults.
        servers = [
            {
                "url": (
                    os.getenv(
                        "SERVER_URL",
                        f"http://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '8000')}",
                    )
                ),
                "description": "Development server",
            },
        ]

        self.app = FastAPI(
            title="Automata MCP Server",
            description="A centralized MCP server using FastAPI with plugin architecture",
            version="1.0.0",
            servers=servers,
        )

        self.api_key = os.getenv("AUTOMATA_API_KEY", "")
        self.host = os.getenv("HOST")
        self.port = os.getenv("PORT")

        self._validate_security_config()

        # Add CORS middleware
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost")
        allowed_origins_list = list(filter(lambda x: x, map(str.strip, allowed_origins.split(","))))
        allowed_methods = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS")
        allowed_methods_list = list(filter(lambda x: x, map(str.strip, allowed_methods.split(","))))
        allowed_headers = os.getenv("ALLOWED_HEADERS", "X-API-Key,Content-Type,Authorization")
        allowed_headers_list = list(filter(lambda x: x, map(str.strip, allowed_headers.split(","))))

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins_list,
            allow_credentials=True,
            allow_methods=allowed_methods_list,
            allow_headers=allowed_headers_list,
        )

        # Add security headers middleware
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            """Add security headers to all responses."""
            response = await call_next(request)

            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            # Remove server header for security
            if "Server" in response.headers:
                del response.headers["Server"]

            return response

        self.tools = {}
        # 配置工具目录路径，支持绝对路径和相对路径
        tools_dir_env = os.getenv("TOOLS_DIR")
        if tools_dir_env is None:
            tools_dir_env = "src"
        if Path(tools_dir_env).is_absolute():
            self.tools_dir = Path(tools_dir_env)
        else:
            self.tools_dir = Path(__file__).parent / tools_dir_env
        self.extension_dir = Path(__file__).parent / "AutoUp-MCP-Extension"
        self.tools_dirs = [self.tools_dir]
        if self.extension_dir.exists() and self.extension_dir.is_dir():
            self.tools_dirs.append(self.extension_dir)
        self.install_dependencies_for_enabled_tools()
        self.discover_tools()
        # Initialize FastApiMCP
        self.mcp = FastApiMCP(self.app)
        self.mcp.mount_http()

        # Include routers
        self.app.include_router(
            create_router(self.authenticate, lambda: len(self.tools), self.tools),
        )

    def _validate_security_config(self):
        """验证安全配置"""
        # 检查API key配置
        if not self.api_key:
            logger.warning(
                "SECURITY: No API key configured. Consider setting AUTOMATA_API_KEY environment variable.",
            )

        # 检查CORS配置
        if "*" in os.getenv("ALLOWED_ORIGINS", "*"):
            logger.warning(
                "SECURITY: CORS allows all origins. Consider restricting ALLOWED_ORIGINS.",
            )

        # 检查调试模式
        if os.getenv("DEBUG", "false").lower() == "true":
            logger.warning(
                "SECURITY: Debug mode is enabled. Sensitive information may be exposed in error responses.",
            )

    @with_exception_handling("dependency_installation")
    def install_dependencies_for_enabled_tools(self):
        """Install dependencies for all tools with improved error handling."""
        logger.info("Installing dependencies for all the enabled tools")
        # 遍历每个工具目录
        for tools_dir in self.tools_dirs:
            try:
                # 检查目录是否是有效目录
                if not tools_dir.is_dir():
                    error_msg = f"Path not exist or not a directory: {tools_dir}"
                    raise ConfigurationError(
                        error_msg,
                        details={"tools_dir": str(tools_dir)},
                    )

                # 遍历工具目录下的每个子目录
                for item in tools_dir.iterdir():
                    if not (item.is_dir() and (item / "__init__.py").exists()):
                        continue

                    modname = item.name
                    try:
                        self._install_single_tool_dependencies(item, modname)
                    except Exception as e:
                        # 继续处理其他工具，但记录错误
                        handle_exception(e, {"tool": modname, "tools_dir": str(tools_dir)})
                        continue
            except Exception as e:
                # 继续处理其他目录，但记录错误
                handle_exception(e, {"tools_dir": str(tools_dir)})

    def _install_single_tool_dependencies(self, tool_dir: Path, modname: str):
        """安装单个工具的依赖"""
        config_path = tool_dir / "config.yaml"
        config = self._load_tool_config(config_path, modname)

        if not config.get("enabled", True):
            logger.info(f"Skipped package installation tool {modname} (tool not enabled)")
            return

        packages = config.get("packages", [])
        if not packages:
            logger.info(f"Skipped package installation for tool {modname} (no packages specified)")
            return

        logger.debug(f"Installing packages for tool {modname}: {packages}")

        try:
            self._run_pip_install(packages, tool_dir.parent.parent)
            logger.info(f"Successfully installed packages for {modname}")
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install packages for {modname}"
            raise DependencyInstallError(
                error_msg,
                details={
                    "tool": modname,
                    "packages": packages,
                    "return_code": e.returncode,
                    "stderr": e.stderr,
                },
            )
        except subprocess.TimeoutExpired:
            error_msg = f"Package installation timed out for {modname}"
            raise DependencyInstallError(
                error_msg,
                details={"tool": modname, "packages": packages},
            )
        except Exception as e:
            error_msg = f"Unexpected error installing packages for {modname}: {e}"
            raise DependencyInstallError(
                error_msg,
                details={"tool": modname, "packages": packages},
            )

    def _load_tool_config(self, config_path: Path, modname: str) -> dict:
        """加载工具配置"""
        if not config_path.exists():
            return {}

        try:
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in config file for tool {modname}"
            raise ConfigurationError(
                error_msg,
                details={
                    "tool": modname,
                    "config_path": str(config_path),
                    "yaml_error": str(e),
                },
            )
        except OSError as e:
            error_msg = f"Cannot read config file for tool {modname}"
            raise ConfigurationError(
                error_msg,
                details={
                    "tool": modname,
                    "config_path": str(config_path),
                    "io_error": str(e),
                },
            )

    def _run_pip_install(self, packages: list[str], cwd: Path):
        """运行 pip 安装命令"""
        # 验证包名安全性（基本检查）
        if not packages:
            return

        for package in packages:
            if not isinstance(package, str) or not package.strip():
                error_msg = f"Invalid package name: {package}"
                raise ValueError(error_msg)

        cmd = ["uv", "pip", "install", *packages]
        # S603: subprocess call is safe because packages are validated above
        subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300,  # 5分钟超时
            check=True,
        )

    @with_exception_handling("tool_discovery")
    def discover_tools(self):
        """Automatically discover tools with improved error handling."""
        for tools_dir in self.tools_dirs:
            try:
                if not tools_dir.is_dir():
                    continue
                for item in tools_dir.iterdir():
                    if not (item.is_dir() and (item / "__init__.py").exists()):
                        continue

                    modname = item.name
                    self._load_and_register_tool(item, modname, tools_dir)
            except Exception as e:
                handle_exception(e, {"tools_dir": str(tools_dir)})

    def _load_and_register_tool(self, tool_dir: Path, modname: str, tools_dir: Path):
        """加载并注册工具"""
        tool_file = tool_dir / f"{modname}_tool.py"
        if not tool_file.exists():
            error_msg = f"Tool file not found: {tool_file}"
            raise ToolLoadError(
                error_msg,
                details={"tool": modname, "expected_file": str(tool_file)},
            )

        try:
            # 计算模块路径
            module_path = self._build_module_path(tools_dir, modname)

            # 导入模块
            try:
                module = importlib.import_module(module_path)
            except Exception as e:
                error_msg = f"Failed to import tool module {module_path}"
                raise ToolLoadError(
                    error_msg,
                    details={
                        "tool": modname,
                        "module_path": module_path,
                        "import_error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

            # 获取工具类
            tool_class_name = self._build_tool_class_name(modname)

            try:
                tool_class = self._get_tool_class(
                    module,
                    tool_class_name,
                    module_path,
                    modname,
                )
            except Exception as e:
                error_msg = f"Failed to get tool class {tool_class_name}"
                raise ToolLoadError(
                    error_msg,
                    details={
                        "tool": modname,
                        "class_name": tool_class_name,
                        "module_path": module_path,
                        "get_class_error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

            try:
                self._validate_tool_class(tool_class, tool_class_name, modname)
            except Exception as e:
                error_msg = f"Failed to validate tool class {tool_class_name}"
                raise ToolLoadError(
                    error_msg,
                    details={
                        "tool": modname,
                        "class_name": tool_class_name,
                        "validation_error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

            # 实例化工具
            try:
                tool_instance = tool_class()
            except Exception as e:
                error_msg = f"Failed to instantiate tool class {tool_class_name}"
                raise ToolLoadError(
                    error_msg,
                    details={
                        "tool": modname,
                        "class_name": tool_class_name,
                        "instantiation_error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

            # 注册工具
            self.tools[modname] = tool_instance

            try:
                self.register_tool_routes(tool_instance, modname)
            except Exception as e:
                error_msg = f"Failed to register routes for tool {modname}"
                raise ToolLoadError(
                    error_msg,
                    details={
                        "tool": modname,
                        "registration_error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

            logger.info(f"Tool {modname} discovered and registered successfully")

        except Exception as e:
            error_msg = f"Unexpected error loading tool {modname}"
            raise ToolLoadError(
                error_msg,
                details={"tool": modname, "error": str(e)},
            )

    def register_tool_routes(self, tool_instance: BaseMCPTool, modname: str):
        """Register FastAPI routes for the tool."""
        route_configs = get_route_configs(tool_instance, modname)
        for config in route_configs:
            self._register_single_route(tool_instance, modname, config)

    def _register_single_route(
        self,
        tool_instance: BaseMCPTool,
        modname: str,
        config: dict,
    ):
        """Register a single tool route."""
        endpoint = config["endpoint"]
        params_class = config["params_class"]
        use_form = config["use_form"]
        tool_name = config["tool_name"]

        # Create endpoint function
        verify_api_key = verify_api_key_dependency(self.authenticate)
        tool_endpoint_func = create_tool_endpoint(
            params_class,
            use_form,
            tool_name,
            tool_instance,
            verify_api_key,
        )

        # Register route
        response_model = tool_instance.get_response_model()
        self.app.post(endpoint, response_model=response_model)(tool_endpoint_func)
        logger.info(f"Registered route {endpoint} for tool {modname}")

    def _build_module_path(self, tools_dir: Path, modname: str) -> str:
        """构建模块路径"""
        app_dir = Path(__file__).parent
        relative_path = tools_dir.relative_to(app_dir)
        return f"app.{relative_path.as_posix().replace('/', '.')}.{modname}"

    def _build_tool_class_name(self, modname: str) -> str:
        """构建工具类名"""
        return "".join(word.capitalize() for word in modname.split("_")) + "Tool"

    def _get_tool_class(
        self,
        module,
        tool_class_name: str,
        module_path: str,
        modname: str,
    ):
        """获取工具类，如果不存在则抛出异常"""
        if not hasattr(module, tool_class_name):
            available_classes = [name for name in dir(module) if name.endswith("Tool")]
            error_msg = (
                f"Tool class {tool_class_name} not found in module {module_path}"
            )
            raise ToolLoadError(
                error_msg,
                details={
                    "tool": modname,
                    "module_path": module_path,
                    "expected_class": tool_class_name,
                    "available_classes": available_classes,
                },
            )
        return getattr(module, tool_class_name)

    def _validate_tool_class(self, tool_class, tool_class_name: str, modname: str):
        """验证工具类是否继承自 BaseMCPTool"""
        if not inspect.isclass(tool_class):
            error_msg = f"Tool {tool_class_name} is not a class"
            raise ToolLoadError(
                error_msg,
                details={
                    "tool": modname,
                    "class_name": tool_class_name,
                    "type": type(tool_class).__name__,
                },
            )

        if not issubclass(tool_class, BaseMCPTool):
            # 获取所有可用的Tool类用于错误信息
            import sys

            current_module = sys.modules.get(tool_class.__module__)
            available_tool_classes = []
            if current_module:
                available_tool_classes = [
                    name
                    for name, obj in vars(current_module).items()
                    if inspect.isclass(obj) and name.endswith("Tool")
                ]

            error_msg = f"Tool class {tool_class_name} must inherit from BaseMCPTool"
            raise ToolLoadError(
                error_msg,
                details={
                    "tool": modname,
                    "class_name": tool_class_name,
                    "required_base": "BaseMCPTool",
                    "available_tool_classes": available_tool_classes,
                },
            )

    def register_tool_routes(self, tool_instance: BaseMCPTool, modname: str):
        """Register FastAPI routes for the tool."""
        route_configs = get_route_configs(tool_instance, modname)
        for config in route_configs:
            self._register_single_route(tool_instance, modname, config)

    def _register_single_route(
        self,
        tool_instance: BaseMCPTool,
        modname: str,
        config: dict,
    ):
        """Register a single tool route."""
        endpoint = config["endpoint"]
        params_class = config["params_class"]
        use_form = config["use_form"]
        tool_name = config["tool_name"]

        # Create endpoint function
        verify_api_key = verify_api_key_dependency(self.authenticate)
        tool_endpoint_func = create_tool_endpoint(
            params_class,
            use_form,
            tool_name,
            tool_instance,
            verify_api_key,
        )

        # Register route
        response_model = tool_instance.get_response_model()
        self.app.post(endpoint, response_model=response_model)(tool_endpoint_func)
        logger.info(f"Registered route {endpoint} for tool {modname}")

    def authenticate(self, api_key: str) -> bool:
        """Authenticate using API key with enhanced security."""
        # If no API key is configured, allow access (development mode)
        if not self.api_key:
            logger.warning("No API key configured - allowing unauthenticated access")
            return True

        # Validate API key format (basic security check)
        if not api_key or not isinstance(api_key, str):
            logger.warning("Invalid API key format")
            return False

        # Check API key length (prevent timing attacks with very short keys)
        if len(api_key.strip()) < 8:
            logger.warning("API key too short")
            return False

        # Use constant-time comparison to prevent timing attacks
        import hmac

        return hmac.compare_digest(api_key.strip(), self.api_key.strip())


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    server = AutomataMCPServer()
    app = server.app

    # 添加全局异常处理器
    @app.exception_handler(AutomataError)
    async def automata_error_handler(_request: Request, exc: AutomataError):
        """处理自定义异常"""
        error_info = exc.to_dict()
        return JSONResponse(
            status_code=400,
            content={"error": error_info},
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """处理未捕获的异常，防止信息泄露"""
        # 记录详细错误信息到日志
        error_info = handle_exception(
            exc,
            {
                "url": str(request.url),
                "method": request.method,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("User-Agent", "Unknown"),
            },
        )

        # 返回安全的错误响应，不包含敏感信息
        safe_error = {
            "error_code": "InternalServerError",
            "message": "An internal server error occurred",
            "timestamp": error_info.get("timestamp", None),
        }

        # 在开发模式下包含更多信息
        if os.getenv("DEBUG", "false").lower() == "true":
            safe_error["details"] = error_info.get("details", {})

        return JSONResponse(
            status_code=500,
            content={"error": safe_error},
        )

    return app


def main():
    """Main entry point for the Automata MCP Server."""

    # 从环境变量读取配置
    load_dotenv()
    host = os.getenv("HOST")
    port = int(os.getenv("PORT"))

    # 启用热重载功能，当代码文件发生变化时自动重启服务器
    uvicorn.run(
        "app.server:create_app",
        host=host,
        port=port,
        factory=True,
        reload=True,
        reload_includes=["**/*.py"],  # 监听所有 Python 文件的变化
    )


if __name__ == "__main__":
    main()
