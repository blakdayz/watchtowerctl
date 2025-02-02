from pip._internal.utils import subprocess
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.result import RunResult
from sentence_transformers import SentenceTransformer
from smolagents import TransformersModel, AgentMemory, CodeAgent
from smolagents.memory import AgentMemory, MemoryStep

from service_manager import ServiceMetadata, InvestigationArtifact


class WorkerAgent(CodeAgent):
    def __init__(self, service: ServiceMetadata):
        super().__init__(service)
        sel


class ServiceManagerAgent(WorkerAgent):
    def __init__(self, embedder_model: str = "TaylorAI/bge-micro-v2"):
        self.embedder = SentenceTransformer(embedder_model)
        super().__init__(
            model="openai:gpt-4o",  # Use your preferred LLM model
            result_type=ServiceMetadata,
            deps_type=None,  # No explicit dependencies for this agent
            system_prompt=self._get_system_prompt(),
        )

    def _get_system_prompt(self) -> str:
        return "You are a macOS Service Manager. You help manage and investigate macOS services."

    #
    def validate_service_metadata(self, ctx: RunContext[None], data: ServiceMetadata) -> ServiceMetadata:
        if not data.label or not data.path:
            raise ModelRetry("Invalid service metadata")
        return data

    async def load_services(self, launchctl_list_path: str = "/usr/bin/launchctl", dyld_dependencies_path: str = "/usr/bin/otool") -> None:
        # Implement the 'load_services' tool using the existing code in LaunchctlManager
        # List Services
        command = [f"{launchctl_list_path}", "print", "system"]
        output = subprocess.call_subprocess(command,show_stdout=False)
        for line in output.splitlines():
            if line.startswith("system"):


    @tool(retries=2)
    async def investigate_service(self, label: str) -> InvestigationArtifact | None:
        # Implement the 'investigate_service' tool using the existing code in LaunchctlManager
        pass

    async def run(self, action: str, **kwargs) -> RunResult[ServiceMetadata]:
        if action.lower() == "loadservices":
            await self.load_services(**kwargs)
            return await super().run("What services have been loaded?", **kwargs)
        elif action.lower() == "investigateservice":
            result = await self.investigate_service(**kwargs)
            if result:
                return await super().run(f"What did you find when investigating service {result.data.label}?", **kwargs)
            else:
                return await super().run("Investigation complete for the specified service.", **kwargs)
        else:
            raise ValueError(f"Unsupported action: {action}")
