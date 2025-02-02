import yara
from models.event_bus import EventBus


def load_yara_rules(rule_path: str, event_bus: EventBus):
    rules = []

    with open(rule_path, "r") as f:
        yara_text = f.read()
        for rule in yara.compile(source=yara_text):
            rules.append(rule)

            # Publish event for each new rule loaded
            event_bus.publish("YARA_RULE_LOADED", rule.metadata["description"])

    return rules
