from asdf.extension import Converter

from StarSharp.datatypes.pointing_model import PointingModel

TAG_POINTING_MODEL = "asdf://starsharp.lsst.io/datatypes/pointing-model-1.0.0"


class PointingModelConverter(Converter):
    tags = [TAG_POINTING_MODEL]
    types = [PointingModel]

    def to_yaml_tree(self, obj, tag, ctx):
        return {
            "schema": obj.schema,
            "matrix": obj.matrix,
        }

    def from_yaml_tree(self, node, tag, ctx):
        return PointingModel(
            schema=node["schema"],
            matrix=node["matrix"],
        )
