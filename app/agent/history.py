"""Keep model tool-call/result groups valid after interrupted turns."""
import json
from functools import wraps


def repair_tool_groups(messages):
    repaired = []
    pending = {}
    def finish():
        for call_id in pending:
            repaired.append({"role": "tool", "tool_call_id": call_id,
                "content": json.dumps({"status": "interrupted", "message": "Result unknown. Verify operation status before retrying."})})
        pending.clear()
    for message in messages:
        if message.get("role") != "tool":
            finish()
        elif message.get("tool_call_id") in pending:
            pending.pop(message["tool_call_id"])
        else:
            continue
        repaired.append(message)
        for call in message.get("tool_calls") or []:
            pending[call["id"]] = True
    finish()
    messages[:] = repaired


def guarded_turn(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        from app.agent.maintenance import drain_maintenance
        await drain_maintenance(self)
        prepare = getattr(getattr(self, "memory_manager", None), "aprepare", None)
        if prepare is not None:
            await prepare(args[0] if args and isinstance(args[0],str) else kwargs.get("user_input",""))
        repair_tool_groups(self.raw_messages)
        try:
            return await method(self, *args, **kwargs)
        except BaseException:
            repair_tool_groups(self.raw_messages)
            raise
    return wrapped


def retain_tool_context(target, sdk_message):
    """Keep provider-required opaque reasoning for tool replay; never emit it to users."""
    value = getattr(sdk_message, 'reasoning_content', None)
    if isinstance(value, str):
        target['_reasoning_content'] = value
    return target
