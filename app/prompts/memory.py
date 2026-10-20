"""Extraction policy retained for documentation; runtime uses verified source extraction."""
STM_EXTRACTION_PROMPT = "仅保留用户明确表达且可定位到原始消息的事实；不推断。已有事实：{existing_facts}"
LTM_EXTRACTION_PROMPT = "必须记录原文、来源、属性、适用范围、观察时间及有效期。推断不得作为事实。已有事实：{existing_ltm}"
