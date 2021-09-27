"""Kedro orchestration entry point."""
# mypy: no-strict-optional
# Global import
from pathlib import Path
from yaml import safe_load
import os
from json import dumps
from kedro.framework.context import KedroContext
from typing import Any, Dict, Union
import logging
from kedro.config import ConfigLoader

# Local import


class DatalabContext(KedroContext):
    """Implement ``KedroContext``."""

    def __init__(
        self,
        package_name: str,
        project_path: Union[Path, str],
        env: str = "local",
        extra_params: Dict[str, Any] = None,
        is_creds_as_envv: bool = False
    ):

        super(DatalabContext, self).__init__(
            package_name=package_name, project_path=project_path, extra_params=extra_params, env=env
        )

        # Load config or provider creds if necessary
        local_env = {}
        provider_path = self.project_path / 'conf' / self.env / 'provider_credentials.json'
        if provider_path.exists():
            local_env["PROVIDER_CREDENTIALS"] = dumps(safe_load(provider_path.open()))

        if is_creds_as_envv:
            conf_paths = ["conf/base", "conf/local"]
            conf_loader = ConfigLoader(conf_paths)
            local_env.update(conf_loader.get("credentials*", "credentials*/**"))

        os.environ.update({k: os.environ.get(k, v) for k, v in local_env.items()})
        logging.info(f"Env variables loaded")
