import os
import json
import yaml
import time
import uuid
import logging
import threading

from pathlib import Path
from typing import Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, MagenticOneGroupChat, Swarm

_AGENT_NAME_MARKER = "----------"

# Default assets dir = agents_manifest/ bundled inside this package
_DEFAULT_ASSETS_DIR = str(Path(__file__).parent / "agents_manifest")
_DEFAULT_MANIFEST_FILE = "default_agents_manifest.yml"
_DEFAULT_TEAM_PRESET = "RoundRobinGroupChat"


class MetadataReviewerCore:
    """Core metadata reviewing logic. Manages sessions, agents, and the AutoGen pipeline."""

    def __init__(self, model_client: Any, assets_dir: str | None = None):
        self.model_client = model_client
        self.assets_dir = assets_dir or _DEFAULT_ASSETS_DIR
        self.sessions: dict = {}
        self._lock = threading.Lock()

    # ── Session management ────────────────────────────────────────────────

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self.sessions[session_id] = {
                "agent_list": None,
                "external_termination": ExternalTermination(),
                "last_used_time": time.time(),
            }
        return session_id

    def get_session(self, session_id: str) -> dict:
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id!r} not found or expired.")
            session = self.sessions[session_id]
            session["last_used_time"] = time.time()
            return session

    def delete_session(self, session_id: str):
        with self._lock:
            self.sessions.pop(session_id, None)

    # ── Model & agent creation ────────────────────────────────────────────

    def load_manifest(self, file_name: str) -> str:
        path = os.path.join(self.assets_dir, file_name)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def list_manifests(self) -> list[str]:
        files = [
            f for f in os.listdir(self.assets_dir)
            if os.path.isfile(os.path.join(self.assets_dir, f))
        ]
        return sorted(files)

    def create_agents(self, agents_manifest: str, session_id: str):
        session = self.get_session(session_id)
        manifest = yaml.safe_load(agents_manifest)
        agent_list = []
        for entry in manifest.get("agents_manifest", []):
            name = entry.get("name")
            system_message = entry.get("system_message")
            if not name or not system_message:
                logging.warning(f"Skipping invalid agent entry: {entry}")
                continue
            agent_list.append(AssistantAgent(
                name=name,
                model_client=self.model_client,
                system_message=system_message,
            ))
        session["agent_list"] = agent_list

    # ── Agent execution ───────────────────────────────────────────────────

    async def start_agents_activity(
        self,
        metadata_to_scan,
        team_preset: str = _DEFAULT_TEAM_PRESET,
        session_id: str = None,
    ):
        """Async generator yielding (streamed_text, None) pairs as agents work."""
        session = self.get_session(session_id)
        agent_list = session["agent_list"]
        text_termination = TextMentionTermination("TERMINATE")
        ext_term = session["external_termination"]

        team_cls = {
            "RoundRobinGroupChat": RoundRobinGroupChat,
            "SelectorGroupChat": SelectorGroupChat,
            "MagenticOneGroupChat": MagenticOneGroupChat,
            "Swarm": Swarm,
        }.get(team_preset, RoundRobinGroupChat)

        team = team_cls(agent_list, termination_condition=text_termination | ext_term, max_turns=len(agent_list))
        await team.reset()

        outputs = []
        async for event in team.run_stream(task=f"Metadata: {metadata_to_scan}"):
            source = getattr(event, "source", None)
            content = getattr(event, "content", None)
            if content:
                outputs.append(
                    f"## **{_AGENT_NAME_MARKER} {source} {_AGENT_NAME_MARKER}**\n\n{content}\n\n"
                )
                yield "".join(outputs), None

    def stop_agents_activity(self, session_id: str):
        self.get_session(session_id)["external_termination"].set()

    # ── JSON extraction ───────────────────────────────────────────────────

    def extract_json(self, text: str):
        text = str(text)
        idx = text.rfind(_AGENT_NAME_MARKER)
        text = text[idx:]
        for opener, closer in (("[", "]"), ("{", "}")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None

    # ── High-level run ────────────────────────────────────────────────────

    async def run(
        self,
        metadata_to_scan,
        manifest_file: str = _DEFAULT_MANIFEST_FILE,
        team_preset: str = _DEFAULT_TEAM_PRESET,
        cancel_flag: threading.Event | None = None,
    ):
        """Run the full pipeline and return the parsed JSON result."""
        session_id = self.create_session()
        try:
            agents_manifest = self.load_manifest(manifest_file)
            self.create_agents(agents_manifest, session_id)

            if cancel_flag and cancel_flag.is_set():
                return None

            final_output = None
            async for output, _ in self.start_agents_activity(metadata_to_scan, team_preset, session_id):
                final_output = output
                if cancel_flag and cancel_flag.is_set():
                    self.stop_agents_activity(session_id)
                    return None

            return self.extract_json(final_output) if final_output else None
        finally:
            self.delete_session(session_id)
