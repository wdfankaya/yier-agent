"""Reply after durable save; run bounded memory maintenance off the reply path."""
import asyncio
from app.config.settings import settings
from app.observability import stage, failure

async def drain_maintenance(agent):
    task = getattr(agent, "_maintenance_task", None)
    if task is not None and task is not asyncio.current_task():
        await asyncio.shield(task)

async def finish_turn(agent, result, save):
    with stage("persist_reply"):
        await asyncio.to_thread(save, agent.session_path, agent.raw_messages, agent.summary,
                                agent.memory_manager.stm_to_dict(), agent.user_id)
    agent._emit("final", reply=result.reply, intent=result.intent.value,
                confidence=result.confidence, requires_human=result.requires_human)

    async def maintain():
        changed = False
        for name, action in (("short_term_memory", lambda: agent.memory_manager.aupdate_short_term(agent.raw_messages[-6:])),
                             ("history_compression", agent._compress_history)):
            if name == "short_term_memory" and not getattr(agent.memory_manager, "memory_enabled", True):
                continue
            if name == "history_compression" and len(agent.raw_messages) <= agent.history_threshold:
                continue
            try:
                with stage(name):
                    await asyncio.wait_for(action(), settings.memory_maintenance_timeout)
                    changed = True
            except Exception as exc:
                failure("maintenance_failed", exc)
        try:
            if changed:
                with stage("persist_maintenance"):
                    await asyncio.to_thread(save, agent.session_path, agent.raw_messages, agent.summary,
                                            agent.memory_manager.stm_to_dict(), agent.user_id)
            await agent._maybe_consolidate()
        except Exception as exc:
            failure("maintenance_save_failed", exc)

    if getattr(agent, "_ephemeral_loop", False):
        await maintain()
    else:
        agent._maintenance_task = asyncio.create_task(maintain())


def tool_managers(agent):
    if hasattr(agent, "tool_manager"):
        return [agent.tool_manager]
    return [sub.tool_manager for sub in getattr(agent, "agents", {}).values()]


def has_pending_work(agent):
    tasks = list(getattr(agent, "_ltm_tasks", ()))
    task = getattr(agent, "_maintenance_task", None)
    if task is not None:
        tasks.append(task)
    return any(not t.done() for t in tasks) or any(getattr(m, "has_pending_work", False) for m in tool_managers(agent))

async def close_resources(agent):
    pending = list(getattr(agent, "_ltm_tasks", ()))
    maintenance = getattr(agent, "_maintenance_task", None)
    if maintenance is not None:
        pending.append(maintenance)
    for task in pending:
        if not task.done():
            task.cancel()
    actions = [asyncio.to_thread(m.close) for m in tool_managers(agent)]
    if hasattr(agent, "client"):
        actions.append(agent.client.close())
    results = await asyncio.gather(*actions, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            failure("resource_close_failed", result)
