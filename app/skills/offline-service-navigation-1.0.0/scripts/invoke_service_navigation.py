from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parents[3]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import vendor_bootstrap  # noqa: F401

from infrastructure.logging.logger import logger
from infrastructure.tools.local.service_station import (
    query_nearest_repair_shops_by_coords,
    resolve_user_location_from_text,
)
from infrastructure.tools.mcp.mcp_servers import baidu_map_mcp


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {"ok": False, "error": "invalid_json", "raw": text}


def _pick_shop(
    shops: list[dict[str, Any]],
    *,
    brand: str | None,
) -> dict[str, Any] | None:
    if not shops:
        return None
    if not brand:
        return shops[0]

    brand = brand.strip().lower()
    for shop in shops:
        haystacks = [
            str(shop.get("service_station_name", "")),
            str(shop.get("supported_brands", "")),
            str(shop.get("service_station_description", "")),
        ]
        joined = " ".join(haystacks).lower()
        if brand in joined:
            return shop
    return shops[0]


def _extract_mcp_text(result: Any) -> str:
    fragments: list[str] = []
    for content in getattr(result, "content", []):
        text = getattr(content, "text", None)
        if text:
            fragments.append(text)
    return "\n".join(fragments).strip()


async def run_service_station(query: str, brand: str | None, limit: int) -> dict[str, Any]:
    location_result = _parse_json(await resolve_user_location_from_text(query))
    if not location_result.get("ok") and location_result.get("source") != "fallback":
        return {
            "ok": False,
            "stage": "resolve_location",
            "detail": location_result,
        }

    shops_result = _parse_json(
        query_nearest_repair_shops_by_coords(
            location_result["lat"],
            location_result["lng"],
            limit,
        )
    )
    shops = shops_result.get("data", [])
    selected = _pick_shop(shops, brand=brand)
    navigation = None

    if selected:
        destination = selected.get("address") or selected.get("service_station_name", "")
        try:
            mcp_result = await baidu_map_mcp.call_tool(
                "map_uri",
                {
                    "service": "direction",
                    "origin": query,
                    "destination": destination,
                },
            )
            navigation = _extract_mcp_text(mcp_result)
        except Exception as exc:
            logger.warning("map_uri failed: %s", exc)
            navigation = f"map_uri failed: {exc}"

    return {
        "ok": True,
        "mode": "service-station",
        "location": location_result,
        "selected_shop": selected,
        "shops": shops,
        "navigation": navigation,
    }


async def run_poi_nav(query: str, destination: str) -> dict[str, Any]:
    location_result = _parse_json(await resolve_user_location_from_text(query))
    if not location_result.get("ok") and location_result.get("source") != "fallback":
        return {
            "ok": False,
            "stage": "resolve_location",
            "detail": location_result,
        }

    try:
        mcp_result = await baidu_map_mcp.call_tool(
            "map_uri",
            {
                "service": "direction",
                "origin": query,
                "destination": destination,
            },
        )
        navigation = _extract_mcp_text(mcp_result)
    except Exception as exc:
        logger.warning("map_uri failed: %s", exc)
        return {
            "ok": False,
            "stage": "map_uri",
            "detail": str(exc),
            "location": location_result,
        }

    return {
        "ok": True,
        "mode": "poi-nav",
        "location": location_result,
        "destination": destination,
        "navigation": navigation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Invoke the offline service/navigation skill."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    service_station = subparsers.add_parser(
        "service-station",
        help="Find nearby service stations and generate navigation info.",
    )
    service_station.add_argument("--query", required=True, help="User location text.")
    service_station.add_argument("--brand", help="Optional brand filter.")
    service_station.add_argument("--limit", type=int, default=3, help="Max shop count.")

    poi_nav = subparsers.add_parser(
        "poi-nav",
        help="Generate navigation for a regular destination.",
    )
    poi_nav.add_argument("--query", required=True, help="User location text.")
    poi_nav.add_argument("--destination", required=True, help="Destination name or address.")

    return parser


async def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "service-station":
        payload = await run_service_station(args.query, args.brand, args.limit)
    else:
        payload = await run_poi_nav(args.query, args.destination)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
