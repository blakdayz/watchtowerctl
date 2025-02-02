import os
import subprocess
from typing import List

from yara import *
from models.event_bus import EventBus


def get_python_site_packages():
    """

    :return:
    """


def get_pip_site_packages():
    return [
        os.path.join(user_site, "lib", "pythonX.Y", "site-packages")
        for user_site in get_python_site_packages()
    ]


def scan_for_malicious_packages(site_packages: List[str], yara_rules):
    malicious_packages = []

    for site_package in site_packages:
        for rule in yara_rules:
            matches = rule.match(site_package)
            if matches:
                malicious_packages.extend(matches)
                break  # No need to check other rules for this package

    return malicious_packages


def check_for_malicious_pip_load(
    yara_rules: List[Rule], threat_level_event_bus: EventBus
):
    site_packages = get_pip_site_packages()
    malicious_packages = scan_for_malicious_packages(site_packages, yara_rules)

    if malicious_packages:
        for package in malicious_packages:
            print(f"Malicious pip package found: {package}")

        # Publish event with the threat level based on the severity of the detected backdoors
        threat_level_event_bus.publish("THREAT_LEVEL_CHANGE", ThreatLevel.HIGH)
