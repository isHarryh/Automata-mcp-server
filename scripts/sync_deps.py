"""
同步 pyproject.toml 和 requirements.txt 的依赖
"""

import re
import sys
from pathlib import Path

from loguru import logger

# 设置输出编码为 UTF-8（Windows 兼容）
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def parse_pyproject_dependencies(pyproject_path: Path) -> list[str]:
    """从 pyproject.toml 解析依赖"""
    content = pyproject_path.read_text(encoding="utf-8")

    # 查找 dependencies 数组
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not match:
        return []

    deps_block = match.group(1)

    # 提取所有依赖项
    dependencies = []
    for raw_line in deps_block.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # 移除引号和逗号
        dep = line.strip('",').strip()
        if dep:
            # 转换格式
            # "fastapi==0.119.1" -> fastapi==0.119.1
            # "pyyaml>=6.0.3" -> pyyaml>=6.0.3
            # "playwright (>=1.55.0,<2.0.0)" -> playwright>=1.55.0,<2.0.0
            dep = dep.replace(" (", "").replace(")", "")
            dependencies.append(dep)

    return sorted(dependencies, key=lambda x: x.lower())


def write_requirements(requirements_path: Path, dependencies: list[str]):
    """写入 requirements.txt"""
    content = "# Auto-generated from pyproject.toml\n"
    content += "# DO NOT EDIT MANUALLY - Run scripts/sync_deps.py instead\n\n"
    content += "\n".join(dependencies) + "\n"

    requirements_path.write_text(content, encoding="utf-8")


def main():
    # 检查是否是 dry-run 模式
    dry_run = "--dry-run" in sys.argv

    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    pyproject_path = project_root / "pyproject.toml"
    requirements_path = project_root / "requirements.txt"

    if not pyproject_path.exists():
        logger.error(f"❌ Error: {pyproject_path} not found")
        sys.exit(1)

    # 解析依赖
    dependencies = parse_pyproject_dependencies(pyproject_path)

    if not dependencies:
        logger.warning("⚠️  Warning: No dependencies found in pyproject.toml")
        sys.exit(1)

    logger.info(f"📦 Found {len(dependencies)} dependencies in pyproject.toml")

    if dry_run:
        logger.info(
            "\n🔍 Dry-run mode - would write the following to requirements.txt:\n",
        )
        for dep in dependencies:
            logger.info(f"  {dep}")
        logger.success("\n✅ Dry-run completed successfully")
        return

    # 写入 requirements.txt
    write_requirements(requirements_path, dependencies)

    logger.success(f"✅ Successfully synced dependencies to {requirements_path}")
    logger.info(f"📝 {len(dependencies)} packages written")


if __name__ == "__main__":
    main()
