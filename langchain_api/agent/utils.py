from datetime import datetime
from pathlib import Path

from deepagents.backends.utils import create_file_data
from langgraph.store.memory import BaseStore
from loguru import logger


def copy_skills_to_store(skills_dir: Path, store: BaseStore):
    """修复版：正确格式存入 store（字典对象）"""
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        logger.error(f"❌ skills 目录不存在: {skills_dir}")
        return

    now = datetime.now().isoformat()
    copied_count = 0
    copy_info = ["\n"]
    for file_path in skills_dir.rglob("*"):
        if "__pycache__" in str(file_path):
            continue
        if file_path.is_file():
            rel_path = file_path.relative_to(skills_dir)
            virtual_path = f"/workspace/skills/{rel_path}"

            try:

                content = file_path.read_text(encoding="utf-8")
                store.put(
                    namespace=("filesystem",),
                    key=virtual_path,
                    value=create_file_data(content),
                )

                copied_count += 1
                copy_info.append(f"已复制: {virtual_path} ({len(content)} chars)")

            except Exception as e:
                copy_info.append(f"复制失败 {virtual_path}: {e}")
    copy_info.append(f"✅ Skills 复制完成: {copied_count} 个文件")
    logger.info("\n".join(copy_info))
