from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from pydantic import BaseModel, Field
import inspect
import os
import sys
import traceback

from open_storyline.config import Settings
from open_storyline.skills.skills_io import dump_skills
from open_storyline.utils.register import NODE_REGISTRY

from open_storyline.mcp.sampling_requester import make_llm
from open_storyline.nodes.core_nodes.base_node import BaseNode
from open_storyline.nodes.node_summary import NodeSummary
from open_storyline.nodes.node_state import NodeState
from src.open_storyline.storage.agent_memory import ArtifactStore

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession


def _find_kokomo_project_root() -> Path | None:
    """
    Discover project root that contains kokomo/config.toml.
    """
    starts: list[Path] = []

    env_root = os.getenv("KOKOMO_PROJECT_ROOT", "").strip()
    if env_root:
        starts.append(Path(env_root).expanduser().resolve())

    starts.append(Path.cwd().resolve())
    starts.append(Path(__file__).resolve())

    for start in starts:
        cursor = start if start.is_dir() else start.parent
        for candidate in [cursor, *cursor.parents]:
            if (candidate / "kokomo" / "config.toml").exists():
                return candidate
    return None


def _resolve_node_catalog_with_kokomo(cfg: Settings) -> tuple[list[str], list[str]]:
    default_pkgs = [str(x) for x in (cfg.local_mcp_server.available_node_pkgs or []) if str(x).strip()]
    default_nodes = [str(x) for x in (cfg.local_mcp_server.available_nodes or []) if str(x).strip()]

    project_root = _find_kokomo_project_root()
    if project_root is None:
        return default_pkgs, default_nodes

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from kokomo.core.runtime_bridge import resolve_node_catalog
    except Exception as e:
        print(f"[Kokomo Bridge] import failed, fallback to OpenStoryline config: {e}")
        return default_pkgs, default_nodes

    cfg_path = Path(
        os.getenv("KOKOMO_CONFIG", str(project_root / "kokomo" / "config.toml"))
    ).expanduser().resolve()

    try:
        contribution, diagnostics = resolve_node_catalog(
            config_path=cfg_path,
            project_root=project_root,
        )
        for item in diagnostics:
            print(f"[Kokomo Bridge] {item}")

        node_pkgs = contribution.node_packages or default_pkgs
        nodes = contribution.available_nodes or default_nodes
        if node_pkgs != default_pkgs or nodes != default_nodes:
            print("[Kokomo Bridge] using Kokomo plugin node catalog")
        return node_pkgs, nodes
    except Exception as e:
        print(f"[Kokomo Bridge] resolve failed, fallback to OpenStoryline config: {e}")
        return default_pkgs, default_nodes


def create_tool_wrapper(node: BaseNode, input_schema: type[BaseModel]):
    """
    Factory function: Convert custom Node to MCP Tool function
    """
    # Get metadata for @server.tool parameters
    meta = node.meta if hasattr(node, 'meta') else None

    async def wrapper(mcp_ctx: Context, **kwargs) -> dict:
        # 1. Unified handling of context and Session
        request = mcp_ctx.request_context.request
        headers = request.headers
        session_id = headers.get('X-Storyline-Session-Id')
        
        # 2. Session lifecycle management
        session_manager = mcp_ctx.request_context.lifespan_context
        if hasattr(session_manager, 'cleanup_expired_sessions'):
            session_manager.cleanup_expired_sessions(session_id)

        # 3. Construct parameters
        # Note: FastMCP automatically injects parameters into kwargs, merge them here
        req_json = await request.json()
        params = kwargs.copy()
        params.update(req_json.get('params', {}).get('arguments', {}))

        node_state = NodeState(
            session_id=session_id,
            artifact_id=params['artifact_id'],
            lang=params.get('lang', 'zh'),
            node_summary=NodeSummary(),
            llm=make_llm(mcp_ctx),
            mcp_ctx=mcp_ctx,
        )
        result = await node(node_state, **params)
        return result


    new_params = []
    # First parameter is fixed as mcp_ctx
    new_params.append(
        inspect.Parameter(
            'mcp_ctx', 
            inspect.Parameter.POSITIONAL_OR_KEYWORD, 
            annotation=Context
        )
    )

    new_annotations = {'mcp_ctx': Context}

    if input_schema:
        for field_name, field_info in input_schema.model_fields.items():
            # Use Annotated to carry description, FastMCP will recognize it as JSON Schema description
            annotation = Annotated[field_info.annotation, field_info]
            
            new_params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=field_info.default if field_info.default is not ... else inspect.Parameter.empty,
                    annotation=annotation
                )
            )
            new_annotations[field_name] = annotation

    wrapper.__name__ = meta.name
    wrapper.__doc__ = meta.description
    wrapper.__signature__ = inspect.Signature(new_params)
    wrapper.__annotations__ = new_annotations
    
    return wrapper, meta


def register(server: FastMCP, cfg: Settings) -> None:
    node_packages, available_nodes = _resolve_node_catalog_with_kokomo(cfg)

    # scan node packages
    for pkg in node_packages:
        NODE_REGISTRY.scan_package(pkg)

    all_node_classes = []
    missing_node_classes = []
    for node_name in available_nodes:
        node_cls = NODE_REGISTRY.get(name=node_name)
        if node_cls is None:
            missing_node_classes.append(node_name)
            continue
        all_node_classes.append(node_cls)

    if missing_node_classes:
        print(f"[Kokomo Bridge] missing node classes: {missing_node_classes}")

    for NodeClass in all_node_classes:
        node_instance = NodeClass(cfg)
        input_schema = node_instance.input_schema

        tool_func, meta = create_tool_wrapper(node_instance, input_schema)
        
        tool_name = NodeClass.meta.name
        tool_desc = NodeClass.meta.description
        
        # Register using server.tool decorator
        server.tool(
            name=tool_name,
            description=tool_desc,
            meta=asdict(meta)
        )(tool_func)

    # Register special tools (e.g., read_node_history)
    # local tool
    @server.tool(
        name="read_node_history",
        description="Retrieve the execution result of any node using its artifact_id"
    )
    async def mcp_read_history(
        mcp_ctx: Context[ServerSession, object],
        query_artifact_id: Annotated[str, Field(description="The artifact_id used to retrieve the corresponding JSON")]
    ) -> dict:
        node_summary = NodeSummary()
        request = mcp_ctx.request_context.request
        session_id = mcp_ctx.request_context.request.headers['X-Storyline-Session-Id']
        req_json_content = await request.json()
        params = req_json_content['params'].get('arguments', {'query_artifact_id': query_artifact_id})
        params['session_id'] = session_id
        params['node_summary'] = node_summary
    
        try:
            store = ArtifactStore(artifacts_dir=params.get('artifacts_dir', ".storyline/.server_cache"), session_id=session_id)
            meta, data = store.load_result(params['query_artifact_id'])
            summary = "History information retrieved successfully"
            isError = False
        except Exception as e:
            traceback_info = ''.join(traceback.format_exception(e))
            summary = f"History read execution failed: {params['query_artifact_id']}\n {traceback_info}",
            meta, data, isError = 'None', 'None', True

        return {
            'artifact_id': params.get('artifact_id', store.generate_artifact_id(req_json_content['params'].get('name', 'read_node_history'))),
            'tool_excute_result': {
                "history": {
                    "meta": meta,
                    "node_data": data,
                }
            },
            'summary': summary,
            'isError': isError,
        }

    @server.tool(
        name="write_skills",
        description="Save the generated Agent Skill (Markdown format) to the file system and return the absolute file path on success."
    )
    async def mcp_write_skills(
        mcp_ctx: Context[ServerSession, object],
        skill_name: Annotated[str, Field(description="Skill file name, e.g., 'fast_paced_vlog', without extension")],
        skill_dir: Annotated[str, Field(description="Skill storage directory, defaults to '.storyline/skills/'")] = '.storyline/skills/',
        skill_content: Annotated[str, Field(description="Skill content in Markdown format")] = '',
    ) -> dict:
        """
        Receives LLM-generated skill content and saves it as a local MD file.
        """
        node_summary = NodeSummary() 
        request = mcp_ctx.request_context.request
        session_id = request.headers.get('X-Storyline-Session-Id', 'unknown_session')
        req_json_content = await request.json()
        params = req_json_content.get('params', {}).get('arguments', {'skill_name': skill_name, 'skill_dir': skill_dir, 'skill_content': skill_content})
        params['session_id'] = session_id
        params['node_summary'] = node_summary
        
        res = await dump_skills(
            skill_name=skill_name,
            skill_dir=skill_dir,
            skill_content=skill_content,
        )
        node_summary.info_for_llm("[Write Skills] Done.")
        store = ArtifactStore(artifacts_dir=params.get('artifacts_dir', ".storyline/.server_cache"), session_id=session_id)

        return {
            'artifact_id': params.get('artifact_id', store.generate_artifact_id(req_json_content.get('params', {}).get('name', 'mcp_write_skills'))),
            'tool_excute_result': {
            },
            'summary': "",
            'isError': False,
        }
        
