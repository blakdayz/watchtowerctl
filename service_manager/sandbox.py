import subprocess
import shutil
import logging

logger = logging.getLogger(__name__)


class Sandbox:
    @staticmethod
    def execute_command(command: str) -> str:
        try:
            return (
                subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Command '{command}' failed with error: {e.output.strip()}")
            raise

    @staticmethod
    def create_sandbox(path: str) -> None:
        try:
            Sandbox.execute_command(f"sandbox -f /dev/null --dryrun /usr/bin/true 1>&2")
            sandbox_profile = f"{path}.sandbox"
            shutil.copyfile(f"/Library/Sandbox/{path}.sandbox", sandbox_profile)
            Sandbox.execute_command(f"sudo mv {sandbox_profile} /Library/Sandbox/")
        except Exception as e:
            logger.error(f"Failed to create sandbox for path '{path}': {e}")
            raise

    @staticmethod
    def delete_sandbox(path: str) -> None:
        try:
            Sandbox.execute_command(f"sudo rm -f '/Library/Sandbox/{path}.sandbox'")
        except Exception as e:
            logger.error(f"Failed to delete sandbox for path '{path}': {e}")
            raise

    @staticmethod
    def disable_sandbox(path: str) -> None:
        try:
            Sandbox.execute_command(
                f"sudo sed -i '' 's/{path}.sandbox//' /Library/Sandbox/ServiceList.plist"
            )
        except Exception as e:
            logger.error(f"Failed to disable sandbox for path '{path}': {e}")
            raise

    @staticmethod
    def enable_sandbox(path: str) -> None:
        try:
            Sandbox.execute_command(
                f"sudo sed -i '' 's//{path}.sandbox/' /Library/Sandbox/ServiceList.plist"
            )
        except Exception as e:
            logger.error(f"Failed to enable sandbox for path '{path}': {e}")
            raise
