import asyncio

from deepagents.middleware.skills import (
    SkillMetadata,
    SkillsMiddleware,
    SkillsStateUpdate,
    _list_skills,
)

from langchain_api.middleware.sandbox.sandbox_system_tool import get_backend


class LangclawSkillsMiddleware(SkillsMiddleware):
    """Langclaw技能中间件"""

    def _get_backend(self, state, runtime, config):
        backend = get_backend(runtime, state)
        return backend

    def before_agent(self, state, runtime, config):
        if "skills_metadata" in state:
            return None

        # Resolve backend (supports both direct instances and factory functions)
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}

        # Load skills from each source in order
        # Later sources override earlier ones (last one wins)
        for source_path in self.sources:
            source_skills = _list_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        # backend.sandbox.kill()
        return SkillsStateUpdate(skills_metadata=skills)

    async def abefore_agent(self, state, runtime, config):
        return await asyncio.to_thread(self.before_agent, state, runtime, config)
