from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.model import ModelSettings
from pydantic_ai.vectordb import Memory
from sentence_transformers import SentenceTransformer


class ServiceManagerAgent(Agent):
    def __init__(self, embedder_model: str = "TaylorAI/bge-micro-v2"):
        self.embedder = SentenceTransformer(embeder_model)
        super().__init__(
            model="openai:gpt-4o",  # Use your preferred LLM model
            result_type=ServiceMetadata,
            deps_type=None,  # No explicit dependencies for this agent
            system_prompt=self._get_system_prompt(),
        )

    def _get_system_prompt(self) -> str:
        return "You are a macOS Service Manager. You help manage and investigate macOS services."

    @result_validator
    def validate_service_metadata(
        self, ctx: RunContext[None], data: ServiceMetadata
    ) -> ServiceMetadata:
        if not data.label or not data.path:
            raise ModelRetry("Invalid service metadata")
        return data

    @tool
    async def load_services(
        self,
        launchctl_list_path: str = "/usr/bin/launchctl",
        dyld_dependencies_path: str = "/usr/bin/otool",
    ) -> None:
        # Implement the 'load_services' tool using the existing code in LaunchctlManager
        pass

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
                return await super().run(
                    f"What did you find when investigating service {result.data.label}?",
                    **kwargs,
                )
            else:
                return await super().run(
                    "Investigation complete for the specified service.", **kwargs
                )
        else:
            raise ValueError(f"Unsupported action: {action}")
