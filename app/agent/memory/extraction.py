"""Compatibility entry points; only source-backed explicit facts are promoted."""
from app.agent.memory.facts import extract, merge

async def extract_short_term_facts(client, model, recent_messages, existing_facts):
    # Plain legacy strings cannot be verified; never promote them to new facts.
    return [f.content for f in merge([], extract(recent_messages, "standalone"), 50)
            if f.active() and not f.requires_refresh]

async def extract_long_term_facts(client, model, messages, summary, existing_facts):
    return extract(messages, "standalone"), ""
