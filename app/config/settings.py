from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目配置，从 .env 文件读取"""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7

    # ReAct 循环
    max_react_steps: int = 5

    # MCP 配置
    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:9123/mcp"

    # RAG 配置
    embedding_model: str = "text-embedding-3-small"
    # Embedding 专用 API 配置（可选）：DeepSeek 等无 embedding 能力的供应商，
    # 需单独指向支持 embeddings 的服务（如硅基流动 https://api.siliconflow.cn/v1）。
    # 留空则复用 openai_api_key / openai_base_url。
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    kb_dir: str = "app/agent/rag/knowledge"
    # 向量后端：numpy（手写余弦，便于调试，无额外向量数据库依赖，默认）/ chroma（向量数据库，需 pip install chromadb）
    #           / pgvector（混合检索=向量 HNSW+全文检索，需 db_enabled=true）
    rag_backend: str = "numpy"
    # NumpyBackend 的 JSON 索引路径
    kb_index_path: str = "app/sessions/kb_index.json"
    # ChromaBackend 的持久化目录与 collection 名
    chroma_persist_dir: str = "app/sessions/chroma"
    chroma_collection: str = "yier_kb"

    # 混合召回 top-N 后再用 cross-encoder 重排。失败回落召回原序。
    # reranker 与 embedding 独立；本仓库 embedding 是 bge-m3/1024，
    # rerank 默认 BAAI/bge-reranker-v2-m3（SiliconFlow /v1/rerank，不重建索引）。
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_recall_k: int = 10
    rerank_api_key: str = ""
    rerank_base_url: str = ""

    # Multi-Agent 配置
    multi_agent_enabled: bool = False

    # DB 持久化配置（PG 落库。db_enabled=True 时 storage/LTM 走 PG，否则回退 JSON）
    db_enabled: bool = False
    database_url: str = (
        "postgresql+asyncpg://postgres:change_me@127.0.0.1:5432/yier_agent"
    )

    # Redis 冷读缓存。热态仍在进程内 Agent dict；Redis 只加速「重启回源」，
    # 不做热路径真相。redis_enabled=False 或 Redis 挂了 → 自动回源 PG，服务不挂。
    redis_enabled: bool = False
    redis_url: str = "redis://127.0.0.1:6379/0"
    session_cache_ttl: int = 1800  # 秒；默认 30 分钟

    # Memory 配置
    memory_enabled: bool = True
    memory_dir: str = "app/sessions/memory"
    memory_user_id: str = "default"
    max_ltm_facts: int = 50
    # 每 N 轮后台巩固一次 LTM（0=只在 close 时巩固）。
    # 强杀进程最多丢这 N 轮；优雅关闭仍会再 consolidate 一次。
    ltm_consolidate_every: int = 3

    # HTTP 按 user_id 令牌桶（10 次/分钟）；LLM 超时/重试；工具超时
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 10
    rate_limit_burst: int = 10
    llm_timeout: float = 60.0
    tool_timeout: float = 10.0
    llm_max_retries: int = 3
    llm_retry_base: float = 1.0  # 退避 1s / 2s / 4s

    # 敏感操作 HITL。False 时 apply_refund 直接执行（评估沙箱关掉，保证可复现）。
    hitl_enabled: bool = True

    # Skill 配置
    skills_enabled: bool = True
    skills_dir: str = "app/agent/skills/definitions"

    # Evaluation 配置（离线评估工具，无聊天开关）
    eval_dataset_path: str = "app/evaluation/cases.json"
    eval_use_judge: bool = True  # 是否启用 LLM-as-judge（质量/幻觉/过程合理性）
    eval_pass_threshold: float = 0.6  # 单维度通过阈值（judge 归一化到 0-1 后比较）

    # 多轮对话管理
    session_path: str = "app/sessions/session.json"
    history_threshold: int = 10  # 原始消息数达到阈值后生成摘要，并保留近期消息
    history_keep_recent: int = 3  # 压缩时保留最近 3 条原始消息

    model_config = {"env_file": ".env"}


settings = Settings()
