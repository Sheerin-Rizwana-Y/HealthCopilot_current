from __future__ import annotations

from datetime import date, datetime
from bson import ObjectId


def serialize(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items() if k != "_id"}
    if isinstance(value, list):
        return [serialize(v) for v in value]
    return value


def serialize_document(value):
    if value is None:
        return None
    result = serialize(value)
    if isinstance(value, dict) and "_id" in value:
        result["id"] = str(value["_id"])
    return result
