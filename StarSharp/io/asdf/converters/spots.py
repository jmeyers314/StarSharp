from asdf.extension import Converter
from astropy.coordinates import Angle

from StarSharp.datatypes.spots import Spots

TAG_SPOTS = "asdf://starsharp.lsst.io/datatypes/spots-1.0.0"


class SpotsConverter(Converter):
    tags = [TAG_SPOTS]
    types = [Spots]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "dx": obj.dx,
            "dy": obj.dy,
            "vignetted": obj.vignetted,
            "field": obj.field,
            "frame": obj.frame,
        }
        if obj.wavelength is not None:
            node["wavelength"] = obj.wavelength
        if obj.rtp is not None:
            node["rtp"] = obj.rtp
        if obj.px is not None:
            node["px"] = obj.px
        if obj.py is not None:
            node["py"] = obj.py
        return node

    def from_yaml_tree(self, node, tag, ctx):
        rtp = node.get("rtp")
        if rtp is not None:
            rtp = Angle(rtp)
        return Spots(
            dx=node["dx"],
            dy=node["dy"],
            vignetted=node["vignetted"],
            field=node["field"],
            frame=node["frame"],
            wavelength=node.get("wavelength"),
            rtp=rtp,
            px=node.get("px"),
            py=node.get("py"),
        )
