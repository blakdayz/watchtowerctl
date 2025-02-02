import asyncio
import subprocess
import sys
from typing import List, Optional

# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def run_command(command: List[str]) -> str:
    """Run a command and return its stdout or stderr if an error occurs."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logging.info(
            f"Command '{' '.join(command)}' succeeded with output: {result.stdout.strip()}"
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Command '{' '.join(command)}' failed with error: {e.stderr.strip()}"
        )
        sys.exit(1)


def parse_suggestions(output: str) -> List[str]:
    """Parse 'did you mean' section from launchctl output and return suggestions."""
    suggestions = []
    lines = output.splitlines()
    in_did_you_mean = False

    for line in lines:
        if "did you mean" in line.lower():
            in_did_you_mean = True
            continue
        elif "usage:" in line.lower():
            break
        if in_did_you_mean and any(
            target in line.lower() for target in ("system/", "pid/", "gui/")
        ):
            suggestions.append(line.strip())

    logging.debug(f"Parsed {len(suggestions)} service suggestions.")
    return suggestions


async def manage_service(service: str, action: str) -> None:
    """Manage a service using launchctl with the given action."""
    logging.info(f"Performing {action.capitalize()} on service: {service}")
    try:
        subprocess.run(["launchctl", action, service], check=True)
        logging.info(f"Successfully {action}ed: {service}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to {action} {service}: {e.stderr.strip()}")


async def mitigate_services(suggestions: List[str]) -> None:
    """Mitigate services by unloading, booting out, and disabling them."""
    if suggestions:
        tasks = [manage_service(service, "unload") for service in suggestions]
        await asyncio.gather(*tasks)

        tasks = [manage_service(service, "bootout") for service in suggestions]
        await asyncio.gather(*tasks)

        tasks = [manage_service(service, "disable") for service in suggestions]
        await asyncio.gather(*tasks)

        # Add 'remove' action to disable task
        tasks = [manage_service(service, "remove") for service in suggestions]
        await asyncio.gather(*tasks)
    else:
        logging.info("No services found in 'did you mean' section.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 launchctl_management.py <filter>")
        sys.exit(1)

    filter_arg = sys.argv[1]
    output = run_command(["launchctl", "print", filter_arg])
    suggestions = parse_suggestions(output)

    asyncio.run(mitigate_services(suggestions))
