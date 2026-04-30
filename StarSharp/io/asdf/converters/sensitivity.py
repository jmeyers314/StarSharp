from asdf.extension import Converter

from StarSharp.datatypes.sensitivity import Sensitivity

TAG_SENSITIVITY = "asdf://starsharp.lsst.io/datatypes/sensitivity-1.0.0"


class SensitivityConverter(Converter):
    tags = [TAG_SENSITIVITY]
    types = [Sensitivity]

    def to_yaml_tree(self, obj, tag, ctx):
        return {
            "gradient": obj.gradient,
            "schema": obj.schema,
            "nominal": obj.nominal,
            "basis": obj.basis,
        }

    def from_yaml_tree(self, node, tag, ctx):
        return Sensitivity(
            gradient=node["gradient"],
            schema=node["schema"],
            nominal=node["nominal"],
            basis=node["basis"],
        )
