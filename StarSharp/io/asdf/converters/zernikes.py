from asdf.extension import Converter
from astropy.coordinates import Angle

from StarSharp.datatypes.double_zernikes import DoubleZernikes
from StarSharp.datatypes.zernikes import Zernikes

TAG_ZERNIKES = "asdf://starsharp.lsst.io/datatypes/zernikes-1.0.0"
TAG_DOUBLE_ZERNIKES = "asdf://starsharp.lsst.io/datatypes/double-zernikes-1.0.0"


class ZernikesConverter(Converter):
    tags = [TAG_ZERNIKES]
    types = [Zernikes]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "coefs": obj.coefs,
            "field": obj.field,
            "frame": obj.frame,
        }
        if obj.R_outer is not None:
            node["R_outer"] = obj.R_outer
        if obj.R_inner is not None:
            node["R_inner"] = obj.R_inner
        if obj.wavelength is not None:
            node["wavelength"] = obj.wavelength
        if obj.rtp is not None:
            node["rtp"] = obj.rtp
        return node

    def from_yaml_tree(self, node, tag, ctx):
        rtp = node.get("rtp")
        if rtp is not None:
            rtp = Angle(rtp)
        return Zernikes(
            coefs=node["coefs"],
            field=node["field"],
            frame=node["frame"],
            R_outer=node.get("R_outer"),
            R_inner=node.get("R_inner"),
            wavelength=node.get("wavelength"),
            rtp=rtp,
        )


class DoubleZernikesConverter(Converter):
    tags = [TAG_DOUBLE_ZERNIKES]
    types = [DoubleZernikes]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "coefs": obj.coefs,
            "field_outer": obj.field_outer,
            "field_inner": obj.field_inner,
            "pupil_outer": obj.pupil_outer,
            "pupil_inner": obj.pupil_inner,
            "frame": obj.frame,
        }
        if obj.wavelength is not None:
            node["wavelength"] = obj.wavelength
        if obj.rtp is not None:
            node["rtp"] = obj.rtp
        return node

    def from_yaml_tree(self, node, tag, ctx):
        rtp = node.get("rtp")
        if rtp is not None:
            rtp = Angle(rtp)
        return DoubleZernikes(
            coefs=node["coefs"],
            field_outer=node["field_outer"],
            field_inner=node["field_inner"],
            pupil_outer=node["pupil_outer"],
            pupil_inner=node["pupil_inner"],
            frame=node["frame"],
            wavelength=node.get("wavelength"),
            rtp=rtp,
        )
