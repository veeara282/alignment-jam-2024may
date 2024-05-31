from jinja2 import Environment, PackageLoader, select_autoescape


JINJA_ENV = Environment(
    loader=PackageLoader("template_utils", "prompts"), autoescape=select_autoescape()
)


class TemplateConsumer:
    def render_template(self, filename: str, **params) -> str:
        template = JINJA_ENV.get_template(filename, globals=vars(self))
        return template.render(**params)
