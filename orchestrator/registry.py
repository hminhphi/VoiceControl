import logging
import os
import re
import time
import json
import httpx
from typing import Any
from utils import read_config

logger = logging.getLogger("orchestrator.registry")

class AgentRegistry:

    def __init__(self, config_path="config/common_config.yaml"):

        logger.info("Initializing AgentRegistry with config: %s", config_path)

        self.config_path = config_path
        self.config = read_config(path=config_path)["agent_registry"]

        self.agent_load_retries = self.config.get("agent_load_retries", 3)
        self.agent_load_retry_delay = self.config.get("agent_load_retry_delay", 2)

        self.agent_card_path = self.config.get(
            "agent_card_path",
            "/.well-known/agent-card.json",
        )

        self.agent_list_path = self.config.get(
            "agent_list_path",
            "config/agent_list.json",
        )

        self._agents = {}
        self._config_agents = {}

        self.reload()

    # ==================================================
    # RELOAD
    # ==================================================

    def reload(self):

        logger.info("Reloading agents from file")

        self._agents.clear()
        self._config_agents.clear()

        self._load_agents_from_file()

    # --------------------------------------------------
    # utils
    # --------------------------------------------------

    def _slug(self, name: str) -> str:
        s = (name or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_") or "agent"

    # --------------------------------------------------

    def _fetch_agent_card(self, url):

        card_url = url.rstrip("/") + self.agent_card_path

        try:

            with httpx.Client(timeout=10) as client:

                r = client.get(card_url)

                if r.status_code == 200:
                    return r.json()

        except Exception as e:

            logger.error("card fetch error %s", e)

        return None

    # --------------------------------------------------

    def _parse_skills(self, card):

        skills = []

        for s in card.get("skills", []):

            skills.append(
                {
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "description": s.get("description"),
                    "examples": s.get("examples", []),
                }
            )

        return skills

    # --------------------------------------------------

    def _parse_orchestrator_description(self, card):

        result = {
            "purpose": "",
            "input_desc": "",
            "output_desc": "",
            "routing_hint": "",
        }

        desc = card.get("description") or ""

        key_to_field = {
            "PURPOSE": "purpose",
            "INPUT": "input_desc",
            "OUTPUT": "output_desc",
            "ROUTING": "routing_hint",
        }

        for key in ("PURPOSE", "INPUT", "OUTPUT", "ROUTING"):

            pattern = re.compile(
                rf"{key}\s*:\s*(.*?)(?=(?:PURPOSE|INPUT|OUTPUT|ROUTING)\s*:|\Z)",
                re.DOTALL | re.IGNORECASE,
            )

            m = pattern.search(desc)

            if m:
                result[key_to_field[key]] = m.group(1).strip()

        return result

    # --------------------------------------------------

    def _input_requires_component_action(self, input_desc):

        s = (input_desc or "").lower()

        return "component" in s and "action" in s

    # ==================================================
    # LOAD CONFIG
    # ==================================================

    def _load_agents_from_file(self):

        if not os.path.isfile(self.agent_list_path):

            logger.warning(
                "Agent list file does not exist: %s",
                self.agent_list_path,
            )

            return

        with open(self.agent_list_path, "r", encoding="utf-8-sig") as f:

            try:
                data = json.load(f)
            except Exception as e:
                logger.error("Error reading agent list: %s", e)
                data = {}

        self._config_agents = data

        for agent_id, info in data.items():

            if not info.get("enabled", True):
                continue

            url = info.get("url")

            if not url:
                continue

            for _ in range(self.agent_load_retries):

                if self._add_runtime(url, agent_id):
                    break

                time.sleep(self.agent_load_retry_delay)

        logger.info(
            "Loaded runtime agents: %d",
            len(self._agents),
        )

    # ==================================================
    # SAVE
    # ==================================================

    def _save_agents_to_file(self):

        with open(self.agent_list_path, "w", encoding="utf-8") as f:

            json.dump(
                self._config_agents,
                f,
                indent=2,
                ensure_ascii=False,
            )

    # ==================================================
    # runtime add
    # ==================================================

    def _add_runtime(self, url, agent_id):

        card = self._fetch_agent_card(url)

        if not card:
            return None

        entry = {
            "agent_id": agent_id,
            "url": url,
            "name": card.get("name"),
            "description": card.get("description", ""),
            "skills": self._parse_skills(card),
        }

        self._agents[agent_id] = entry

        return entry

    # ==================================================
    # PUBLIC
    # ==================================================

    def get_all(self):

        # IMPORTANT: reload every time
        self.reload()

        return list(self._agents.values())

    def get(self, agent_id):

        self.reload()

        return self._agents.get(agent_id)

    def get_catalog(self):

        self.reload()

        return self._config_agents


    def get_url(self, agent_id):

        agent = self._agents.get(agent_id)

        return agent["url"] if agent else None


    # -------------------------
    # add
    # -------------------------

    def add(self, url, agent_id):

        self.reload()

        self._config_agents[agent_id] = {
            "url": url,
            "enabled": True,
        }

        self._save_agents_to_file()

        self.reload()

        return self._agents.get(agent_id)


    # -------------------------
    # remove = disable
    # -------------------------

    def remove(self, agent_id):

        self.reload()

        if agent_id not in self._config_agents:
            return False

        self._config_agents[agent_id]["enabled"] = False

        self._save_agents_to_file()

        self.reload()

        return True


if __name__ == "__main__":
    
    registry = AgentRegistry()
    print("Loaded agents:")
    for agent in registry.get_all():
        print(f"- {agent['agent_id']}: {agent['url']} (name: {agent['name']})")
    
    # Example: Add a new agent (simulate with a dummy URL)
    print("\nAdding a new agent (should fail if URL not responding):")
    new_url = "http://example_agent:9000"
    agent_id = "example_agent"
    added = registry.add(new_url, agent_id)
    if added:
        print(f"Added agent: {added['agent_id']}")
    else:
        print("Failed to add new agent (expected if dummy URL)")

    # Example: Remove an agent
    print("\nRemoving 'cloud' agent:")
    if registry.remove("cloud"):
        print("Removed 'cloud' agent successfully.")
    else:
        print("Cloud agent not found or could not be removed.")
