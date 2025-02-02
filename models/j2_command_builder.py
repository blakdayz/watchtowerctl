import jinja2
from jinja2.loaders import FileSystemLoader
from jinja2.exceptions import TemplateNotFound

from runtime_stability import logging


class J2CommandBuilder:
    def __init__(self):
        self.env = jinja2.Environment(
            loader=FileSystemLoader("feature_steps"),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=True,
        )

    def generate_socat_dump_command(
        self, address: str, port: int, output_file: str
    ) -> str:
        template = self.env.get_template("socat_dump.j2")
        return template.render(address=address, port=port, output_file=output_file)

    def generate_socat_sinkhole_command(
        self, address: str, port: int, sinkhole_address: str, sinkhole_port: int
    ) -> str:
        try:
            template = self.env.get_template("socat_sinkhole.j2")
            return template.render(
                address=address,
                port=port,
                sinkhole_address=sinkhole_address,
                sinkhole_port=sinkhole_port,
            )

        except TemplateNotFound as e:
            logging.error(e)
