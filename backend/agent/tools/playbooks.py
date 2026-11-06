"""Load local markdown playbooks (Firecrawl-style skills, no Firecrawl API)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class NoPlaybookArgs(BaseModel):
    pass

from ..playbook_store import PlaybookStore
from ..types import Risk
from .registry import ToolContext, tool

_store = PlaybookStore()


class PlaybookNameArgs(BaseModel):
    name: str = Field(min_length=1, description="Playbook name from list_playbooks")


class PlaybookUrlArgs(BaseModel):
    url: str = Field(min_length=1, description="Page URL or domain")


@tool(
    name="list_playbooks",
    description="List local playbooks for hard pages, e-commerce, extraction, and known sites.",
    params=NoPlaybookArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def list_playbooks(args: NoPlaybookArgs, ctx: ToolContext) -> str:
    return _store.catalog()


@tool(
    name="load_playbook",
    description="Load full instructions for a named playbook.",
    params=PlaybookNameArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def load_playbook(args: PlaybookNameArgs, ctx: ToolContext) -> str:
    pb = _store.get(args.name)
    if pb is None:
        return f"Unknown playbook {args.name!r}. { _store.catalog()}"
    return f"# {pb.name}\n{pb.description}\n\n{pb.body}"


@tool(
    name="lookup_site_playbook",
    description="Load a site-specific playbook if one exists for this URL.",
    params=PlaybookUrlArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def lookup_site_playbook(args: PlaybookUrlArgs, ctx: ToolContext) -> str:
    pb = _store.for_url(args.url)
    if pb is None:
        return f"No site playbook for {args.url}"
    return f"# {pb.name}\n{pb.body}"
