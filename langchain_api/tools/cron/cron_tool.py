from typing import Annotated, Literal, Optional

from langchain_core.tools import tool

from langchain_api.tools.cron.cron_manager import get_cron_manager


@tool
def cron_tool(
    action: Annotated[
        Literal["add", "list", "remove"],
        "The action to perform: add a new cron job, list existing jobs, or remove a job",
    ],
    name: Annotated[
        Optional[str], "Unique name for the cron job (required for add/remove)"
    ] = None,
    cron_expression: Annotated[
        Optional[str],
        "Cron expression for add action. Format: 'minute hour day month weekday'. Examples: '0 9 * * *' (daily at 9am), '*/15 * * * *' (every 15 min), '0 0 * * 0' (weekly)",
    ] = None,
    command: Annotated[Optional[str], "Command to execute (required for add)"] = None,
    description: Annotated[Optional[str], "Optional description for add action"] = None,
    job_id: Annotated[Optional[int], "Job ID for remove"] = None,
    enabled_only: Annotated[Optional[bool], "Filter by enabled status for list"] = True,
) -> str:
    """
    Manage cron jobs: add, list, or remove scheduled tasks.

    Examples:
        - Add a daily job at 9am: action="add", name="daily_backup", cron_expression="0 9 * * *", command="python backup.py"
        - List all jobs: action="list"
        - List only enabled jobs: action="list", enabled_only=true
        - Remove by name: action="remove", name="daily_backup"
        - Remove by id: action="remove", job_id=1
    """
    manager = get_cron_manager()

    if action == "add":
        if not name or not cron_expression or not command:
            return (
                "Error: name, cron_expression, and command are required for add action"
            )
        try:
            job = manager.add(
                name=name,
                cron_expression=cron_expression,
                command=command,
                description=description,
            )
            return f"Cron job '{job.name}' added successfully (ID: {job.id})"
        except Exception as e:
            return f"Failed to add cron job: {e}"

    elif action == "list":
        jobs = manager.list(enabled=enabled_only)
        if not jobs:
            return "No cron jobs found."
        lines = []
        for job in jobs:
            status = "enabled" if job.enabled else "disabled"
            lines.append(
                f"[{job.id}] {job.name} ({status})\n"
                f"  cron: {job.cron_expression}\n"
                f"  command: {job.command}\n"
                f"  description: {job.description or 'N/A'}\n"
            )
        return "\n".join(lines)

    elif action == "remove":
        if job_id is None and name is None:
            return "Error: Please provide either job_id or name"
        success = manager.remove(job_id=job_id, name=name)
        return "Cron job removed successfully" if success else "Cron job not found"

    return "Unknown action"


if __name__ == "__main__":
    print(cron_tool.invoke({"action": "add", "name": "daily_report", "cron_expression": "0 9 * * *", "command": "python scripts/report.py", "description": "Daily report"}))
    # print(cron_tool.invoke({"action": "list"}))
    # print(cron_tool.invoke({"action": "remove", "name": "daily_report"}))
