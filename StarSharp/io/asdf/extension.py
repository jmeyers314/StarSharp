from asdf.extension import Extension

from .converters.field_coords import FieldCoordsConverter
from .converters.moments import MomentsConverter
from .converters.pointing_model import PointingModelConverter
from .converters.sensitivity import SensitivityConverter
from .converters.spots import SpotsConverter
from .converters.state import StateConverter, StateFactoryConverter, StateSchemaConverter
from .converters.zernikes import DoubleZernikesConverter, ZernikesConverter

EXTENSION_URI = "asdf://starsharp.lsst.io/extensions/starsharp-1.0.0"


class StarSharpExtension(Extension):
    extension_uri = EXTENSION_URI
    converters = [
        FieldCoordsConverter(),
        StateSchemaConverter(),
        StateConverter(),
        StateFactoryConverter(),
        ZernikesConverter(),
        DoubleZernikesConverter(),
        SpotsConverter(),
        MomentsConverter(),
        SensitivityConverter(),
        PointingModelConverter(),
    ]
    tags = [
        "asdf://starsharp.lsst.io/datatypes/field-coords-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/state-schema-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/state-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/state-factory-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/zernikes-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/double-zernikes-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/spots-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/moments-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/sensitivity-1.0.0",
        "asdf://starsharp.lsst.io/datatypes/pointing-model-1.0.0",
    ]
