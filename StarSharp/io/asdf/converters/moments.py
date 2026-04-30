from asdf.extension import Converter
from astropy.coordinates import Angle

from StarSharp.datatypes.moments import Moments, Moments2, Moments3, Moments4

TAG_MOMENTS = "asdf://starsharp.lsst.io/datatypes/moments-1.0.0"


class MomentsConverter(Converter):
    tags = [TAG_MOMENTS]
    types = [Moments, Moments2, Moments3, Moments4]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "order": obj.order,
            "values": obj.values,
            "frame": obj.frame,
        }
        if obj.field is not None:
            node["field"] = obj.field
        if obj.rtp is not None:
            node["rtp"] = obj.rtp
        return node

    def from_yaml_tree(self, node, tag, ctx):
        rtp = node.get("rtp")
        if rtp is not None:
            rtp = Angle(rtp)
        order = node["order"]
        return Moments[order]._create(
            order=order,
            values=node["values"],
            frame=node["frame"],
            field=node.get("field"),
            rtp=rtp,
        )
