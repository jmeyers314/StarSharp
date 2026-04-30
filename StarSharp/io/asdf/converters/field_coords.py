from asdf.extension import Converter
from astropy.coordinates import Angle

from StarSharp.datatypes.field_coords import FieldCoords

TAG = "asdf://starsharp.lsst.io/datatypes/field-coords-1.0.0"


class FieldCoordsConverter(Converter):
    tags = [TAG]
    types = [FieldCoords]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "x": obj.x,
            "y": obj.y,
            "frame": obj.frame,
        }
        if obj.rtp is not None:
            node["rtp"] = obj.rtp
        return node

    def from_yaml_tree(self, node, tag, ctx):
        rtp = node.get("rtp")
        if rtp is not None:
            rtp = Angle(rtp)
        return FieldCoords(
            x=node["x"],
            y=node["y"],
            frame=node["frame"],
            rtp=rtp,
        )
