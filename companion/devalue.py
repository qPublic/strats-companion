"""Minimal port of the `devalue` wire format that the Strats.gg API uses.

Only the subset that shows up in lineup data is supported: JSON types,
`undefined`, Dates (kept as ISO strings), Sets and Maps.
"""

import json

UNDEFINED = -1
HOLE = -2
NAN = -3
POSITIVE_INFINITY = -4
NEGATIVE_INFINITY = -5
NEGATIVE_ZERO = -6

_SPECIALS = {
    UNDEFINED: None,
    HOLE: None,
    NAN: float("nan"),
    POSITIVE_INFINITY: float("inf"),
    NEGATIVE_INFINITY: float("-inf"),
    NEGATIVE_ZERO: -0.0,
}


def parse(serialized):
    return unflatten(json.loads(serialized))


def unflatten(parsed):
    if isinstance(parsed, int) and not isinstance(parsed, bool):
        return _SPECIALS[parsed]
    values = parsed
    hydrated = {}

    def hydrate(index):
        if index < 0:
            return _SPECIALS[index]
        if index in hydrated:
            return hydrated[index]
        value = values[index]
        if isinstance(value, list):
            if value and isinstance(value[0], str):
                result = _hydrate_typed(value)
            else:
                result = []
                hydrated[index] = result
                result.extend(hydrate(i) for i in value)
        elif isinstance(value, dict):
            result = {}
            hydrated[index] = result
            for key, i in value.items():
                result[key] = hydrate(i)
        else:
            result = value
        hydrated[index] = result
        return result

    def _hydrate_typed(value):
        kind = value[0]
        if kind == "Date":
            return value[1]
        if kind == "Set":
            return [hydrate(i) for i in value[1:]]
        if kind == "Map":
            return {hydrate(value[i]): hydrate(value[i + 1]) for i in range(1, len(value), 2)}
        if kind == "null":
            return {value[i]: hydrate(value[i + 1]) for i in range(1, len(value), 2)}
        if kind == "BigInt":
            return int(value[1])
        if kind == "Object":
            return value[1]
        raise ValueError(f"Unsupported devalue type: {kind}")

    return hydrate(0)


def stringify(value):
    if value is None:
        return str(UNDEFINED)
    flat = []
    indexes = {}

    def flatten(thing):
        if thing is None:
            return UNDEFINED
        key = (type(thing).__name__, thing) if isinstance(thing, (str, int, float, bool)) else None
        if key is not None and key in indexes:
            return indexes[key]
        index = len(flat)
        flat.append(None)
        if key is not None:
            indexes[key] = index
        if isinstance(thing, dict):
            flat[index] = {k: flatten(v) for k, v in thing.items()}
        elif isinstance(thing, (list, tuple)):
            flat[index] = [flatten(v) for v in thing]
        else:
            flat[index] = thing
        return index

    flatten(value)
    return json.dumps(flat, separators=(",", ":"))
