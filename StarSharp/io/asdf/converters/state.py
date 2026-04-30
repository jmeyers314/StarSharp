import astropy.units as u
from asdf.extension import Converter

from StarSharp.datatypes.state import State, StateFactory, StateSchema

TAG_SCHEMA = "asdf://starsharp.lsst.io/datatypes/state-schema-1.0.0"
TAG_STATE = "asdf://starsharp.lsst.io/datatypes/state-1.0.0"
TAG_FACTORY = "asdf://starsharp.lsst.io/datatypes/state-factory-1.0.0"


class StateSchemaConverter(Converter):
    tags = [TAG_SCHEMA]
    types = [StateSchema]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "dof_names": list(obj.dof_names),
            "dof_units": [unit.to_string() for unit in obj.dof_units],
            "use_dof": obj.use_dof,  # always a numpy array after __post_init__
        }
        if obj.step is not None:
            node["step"] = obj.step
        if obj.Vh is not None:
            node["Vh"] = obj.Vh
        if obj.S is not None:
            node["S"] = obj.S
        if obj.U is not None:
            node["U"] = obj.U
        return node

    def from_yaml_tree(self, node, tag, ctx):
        return StateSchema(
            dof_names=tuple(node["dof_names"]),
            dof_units=tuple(u.Unit(s) for s in node["dof_units"]),
            use_dof=node["use_dof"],
            step=node.get("step"),
            Vh=node.get("Vh"),
            S=node.get("S"),
            U=node.get("U"),
        )


class StateConverter(Converter):
    tags = [TAG_STATE]
    types = [State]

    def to_yaml_tree(self, obj, tag, ctx):
        return {
            "value": obj.value,
            "schema": obj.schema,
            "basis": obj.basis,
        }

    def from_yaml_tree(self, node, tag, ctx):
        return State(
            value=node["value"],
            schema=node["schema"],
            basis=node["basis"],
        )


class StateFactoryConverter(Converter):
    tags = [TAG_FACTORY]
    types = [StateFactory]

    def to_yaml_tree(self, obj, tag, ctx):
        return {"schema": obj.schema}

    def from_yaml_tree(self, node, tag, ctx):
        return StateFactory(schema=node["schema"])
