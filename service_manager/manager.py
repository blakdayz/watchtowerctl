import os
import subprocess
import shutil
import logging
import logging.config
import time
from typing import List, Dict, Any
import concurrent.futures
from lxml import html
from bs4 import BeautifulSoup

from . import ServiceManagerError

# Load logging configuration
logging.config.fileConfig("service_manager/logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("service_manager")

from .output_models import ServiceMetadata, InvestigationArtifact
from sandbox import
from search import SearchManager
from utils import init_sentence_transformer_model
from vectordb_manager import VectorDBManager

class Service:
    def __init__(self, label: str, pid: int, path: str):
        self.label = label
        self.pid = pid
        self.path = path
        self.is_malicious = False
        self.metadata = None
        self.embedding = None

    def set_embedding(self, embedding: List[float]):
        self.embedding = embedding

class LaunchctlManager:
    def __init__(self, config: tuple):
        self.milvus_config, self.llm_model, self.embedder_model, self.vectordb_embeddings = config
        self.services: Dict[str, Service] = {}
        self.vector_db_manager = VectorDBManager(self.vectordb_embeddings)
        self.embedder = init_sentence_transformer_model(self.embedder_model)
        self.man_page_retriever = ManPageRetriever(self.llm_model)
        self.sandbox = Sandbox()
        self.search_manager = SearchManager(self.vector_db_manager.memory)
        self.load_services()

    def _get_launchctl_list(self) -> List[str]:
        try:
            output = subprocess.check_output("launchctl list", shell=True).decode().split("\n")[1:-1]
            return [line.split()[0] for line in output]
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get launchctl list: {e}")
            raise ServiceManagerError("Failed to get launchctl list") from e

    def _get_service_path(self, label: str) -> str:
        try:
            return subprocess.check_output(f"launchctl load {label}", shell=True).decode().strip().split(":")[-1].strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get service path for label '{label}': {e}")
            raise ServiceManagerError(f"Failed to get service path for label '{label}'") from e

    def load_services(self):
        try:
            self._get_launchctl_list()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._process_service, label, Service(label, -1, "")) for label in self.services.keys()}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        except Exception as e:
            logger.error(f"Failed to load services: {e}")
            raise ServiceManagerError("Failed to load services") from e

    def _process_service(self, label: str, service: Service) -> None:
        try:
            path = self._get_service_path(label)
            if not path:
                return

            service.path = path

            metadata = {
                "label": label,
                "path": path,
            }
            service.metadata = metadata

            text = f"{metadata['label']} {path} {' '.join(self._get_dyld_dependencies(path))}"
            embedding = self.embedder.encode([text])[0].tolist()
            service.set_embedding(embedding)

            self.vector_db_manager.save([text], [metadata])
        except Exception as e:
            logger.error(f"Failed to process service {label}: {e}")
            raise ServiceManagerError(f"Failed to process service {label}") from e

    def _get_dyld_dependencies(self, path: str) -> List[str]:
        try:
            output = subprocess.check_output(f"otool -L '{path}'", shell=True).decode().split("\n")[1:-1]
            return [line.split()[-2] for line in output]
        except (subprocess.CalledProcessError, IndexError) as e:
            logger.error(f"Failed to get dyld dependencies for path '{path}': {e}")
            raise ServiceManagerError(f"Failed to get dyld dependencies for path '{path}'") from e

    def identify_non_standards(self):
        try:
            standard_services = self._get_standard_services()
            non_std_services = [label for label in self.services.keys() if label not in standard_services]
            for label in non_std_services:
                logger.warning(f"Non-standard service found: {label}")
        except Exception as e:
            logger.error(f"Failed to identify non-standard services: {e}")
            raise ServiceManagerError("Failed to identify non-standard services") from e

    def _get_standard_services(self) -> List[str]:
        """
        Currently we are working in hostile until proven can behave mindset
        We need to log the behavior of a process. Is it visiting alot of other threads?
        Does it offer generic symbolesa referencing weakrefs on bound objects with a wrapping/contexst
        pattern? If so, we want to keep a threat watch list ;
        :return:
        """
        return ["com.apple.*.plist"]

    def monitor_services(self):
        while True:
            try:
                self.load_services()
                self.identify_non_standards()
            except Exception as e:
                logger.error(f"Failed to monitor services: {e}")
                raise ServiceManagerError("Failed to monitor services") from e
            time.sleep(60)  # Monitor every minute (adjust as needed)

   async def search_services(self, query: str) -> List[Dict[str, Any]]:
        try:
            results = self.vector_db_manager.memory.search([query], top_n=50, unique=True, batch_results="diverse")
            return [result['metadata'] for result in results]
        except Exception as e:
            logger.error(f"Failed to search services: {e}")
            raise ServiceManagerError("Failed to search services") from e
