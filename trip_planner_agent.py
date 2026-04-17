from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pydeck as pdk
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ------------------------------------------------------------
# App constants
# ------------------------------------------------------------
APP_NAME = "Voyager Agent"
DATA_DIR = Path("data")
STATE_FILE = DATA_DIR / "voyager_state.json"
VOTES_FILE = DATA_DIR / "poi_votes.jsonl"

NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
WIKIVOYAGE_ENDPOINT = "https://en.wikivoyage.org/w/api.php"

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
LIGHT_MAP = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
DARK_MAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

VALID_INTERESTS = [
    "outdoors",
    "food",
    "coffee",
    "museums",
    "history",
    "art",
    "nightlife",
    "scenic",
]

INTEREST_TAGS: Dict[str, List[Tuple[str, str]]] = {
    "outdoors": [("leisure", "park|garden|nature_reserve"), ("tourism", "viewpoint"), ("natural", "peak|wood|spring|beach|cave_entrance")],
    "food": [("amenity", "restaurant|fast_food|food_court")],
    "coffee": [("amenity", "cafe")],
    "museums": [("tourism", "museum|gallery")],
    "history": [("historic", ".+"), ("tourism", "attraction")],
    "art": [("tourism", "gallery|museum|artwork")],
    "nightlife": [("amenity", "bar|pub|nightclub")],
    "scenic": [("tourism", "viewpoint"), ("natural", "peak|beach")],
}

BLOCK_NAMES = ["morning", "afternoon", "evening"]
FALLBACK_THEMES = {
    "relaxed": {"morning": 1, "afternoon": 1, "evening": 1},
    "balanced": {"morning": 1, "afternoon": 2, "evening": 1},
    "packed": {"morning": 2, "afternoon": 2, "evening": 2},
}


# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class AppSnapshot:
    itinerary: Optional[Dict[str, Any]]
    poi_catalog: Dict[str, Dict[str, Any]]
    map_center: Dict[str, Optional[float]]
    city_key: str


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------
def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def safe_json_load(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def write_json(path: Path, payload: Any) -> None:
    ensure_storage()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_user_agent(contact: str) -> str:
    cleaned = (contact or "").strip()
    if not cleaned:
        cleaned = "contact@example.com"
    return f"voyager-agent/1.0 ({cleaned})"


def trace(kind: str, **payload: Any) -> None:
    st.session_state.setdefault("run_trace", []).append({"time": time.time(), "kind": kind, **payload})


def reset_trace() -> None:
    st.session_state["run_trace"] = []


def render_trace() -> None:
    with st.expander("Execution trace", expanded=False):
        events = st.session_state.get("run_trace", [])
        if not events:
            st.caption("No steps yet.")
            return
        for item in events:
            t = time.strftime("%H:%M:%S", time.localtime(item["time"]))
            kind = item["kind"]
            if kind == "model":
                st.markdown(f"**{t}** model step {item.get('step')}")
            elif kind == "tool":
                st.markdown(f"**{t}** tool `{item.get('name')}`")
                if item.get("arguments") is not None:
                    st.code(json.dumps(item["arguments"], indent=2), language="json")
            elif kind == "tool_result":
                st.markdown(f"**{t}** result `{item.get('name')}` in {item.get('elapsed')}s")
            elif kind == "note":
                st.markdown(f"**{t}** {item.get('message')}")
            elif kind == "error":
                st.markdown(f"**{t}** error `{item.get('where')}`")
                st.code(str(item.get("message", "")))


# ------------------------------------------------------------
# Persistence
# ------------------------------------------------------------
def save_snapshot() -> None:
    if not st.session_state.get("autosave_enabled", True):
        return
    itinerary = st.session_state.get("itinerary")
    catalog = st.session_state.get("poi_catalog") or {}
    if not itinerary or not catalog:
        return
    payload = {
        "itinerary": itinerary,
        "poi_catalog": catalog,
        "map_center": st.session_state.get("map_center", {}),
        "city_key": st.session_state.get("city_key", ""),
    }
    write_json(STATE_FILE, payload)


def restore_snapshot() -> None:
    if st.session_state.get("_snapshot_restored"):
        return
    st.session_state["_snapshot_restored"] = True
    payload = safe_json_load(STATE_FILE, {})
    if not isinstance(payload, dict):
        return
    if payload.get("itinerary") and payload.get("poi_catalog"):
        st.session_state["itinerary"] = payload.get("itinerary")
        st.session_state["poi_catalog"] = payload.get("poi_catalog", {})
        st.session_state["map_center"] = payload.get("map_center", {})
        st.session_state["city_key"] = payload.get("city_key", "")


def clear_snapshot() -> None:
    for key in ["itinerary", "poi_catalog", "map_center", "city_key"]:
        st.session_state.pop(key, None)
    if STATE_FILE.exists():
        try:
            STATE_FILE.unlink()
        except Exception:
            pass


# ------------------------------------------------------------
# Feedback memory
# ------------------------------------------------------------
def append_vote(city_key: str, poi_id: str, vote: str) -> None:
    ensure_storage()
    with VOTES_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"city_key": city_key, "poi_id": poi_id, "vote": vote, "time": time.time()}, ensure_ascii=False) + "\n")


def vote_adjustments(city_key: str) -> Dict[str, float]:
    if not VOTES_FILE.exists():
        return {}
    scores: Dict[str, float] = {}
    try:
        for line in VOTES_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("city_key") != city_key:
                continue
            poi_id = row.get("poi_id")
            if not poi_id:
                continue
            delta = 0.25 if row.get("vote") == "up" else -0.35 if row.get("vote") == "down" else 0.0
            scores[poi_id] = scores.get(poi_id, 0.0) + delta
    except Exception:
        return {}
    return scores


# ------------------------------------------------------------
# OSM integration
# ------------------------------------------------------------
def get_json(url: str, *, params: Dict[str, Any], headers: Dict[str, str], timeout: int = 20) -> Any:
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60 * 60 * 24)
def geocode_place(place_name: str, contact: str) -> Optional[Dict[str, Any]]:
    headers = {"User-Agent": make_user_agent(contact)}
    params = {"q": place_name, "format": "json", "limit": 1}
    time.sleep(1.0)
    payload = get_json(NOMINATIM_ENDPOINT, params=params, headers=headers)
    if not payload:
        return None
    top = payload[0]
    return {
        "name": top.get("display_name", place_name),
        "lat": float(top["lat"]),
        "lon": float(top["lon"]),
    }


def combine_tag_rules(interests: Iterable[str]) -> Dict[str, str]:
    merged: Dict[str, List[str]] = {}
    selected = [tag for interest in interests for tag in INTEREST_TAGS.get(interest, [])]
    if not selected:
        selected = [
            ("tourism", "attraction|museum|viewpoint"),
            ("leisure", "park|garden"),
            ("amenity", "cafe|restaurant"),
            ("historic", ".+"),
        ]
    for key, value in selected:
        merged.setdefault(key, []).append(value)
    return {key: "|".join(values) for key, values in merged.items()}



def build_overpass_query(lat: float, lon: float, radius_m: int, rule_map: Dict[str, str]) -> str:
    sections: List[str] = []
    for key, value in rule_map.items():
        sections.append(f'node(around:{radius_m},{lat},{lon})["{key}"~"{value}"];')
        sections.append(f'way(around:{radius_m},{lat},{lon})["{key}"~"{value}"];')
        sections.append(f'relation(around:{radius_m},{lat},{lon})["{key}"~"{value}"];')
    body = "\n".join(sections)
    return f"""
[out:json][timeout:35];
(
{body}
);
out center tags;
"""



def infer_category(tags: Dict[str, str]) -> str:
    for key in ["tourism", "amenity", "historic", "leisure", "natural"]:
        if key in tags:
            return f"{key}:{tags[key]}"
    return "other"



def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6371.0
    dlat = math.radians(b_lat - a_lat)
    dlon = math.radians(b_lon - a_lon)
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    c = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(c))


@st.cache_data(ttl=60 * 60 * 6)
def discover_pois(city: str, interests: Tuple[str, ...], radius_km: float, limit: int, contact: str) -> Dict[str, Any]:
    geo = geocode_place(city, contact)
    if not geo:
        return {"city_key": city.strip().lower(), "display_name": city, "center": {"lat": None, "lon": None}, "pois": [], "error": "Could not geocode city."}

    query = build_overpass_query(
        geo["lat"],
        geo["lon"],
        max(500, int(radius_km * 1000)),
        combine_tag_rules(interests),
    )

    headers = {"User-Agent": make_user_agent(contact)}
    failure: Optional[str] = None
    for attempt in range(3):
        try:
            response = requests.post(OVERPASS_ENDPOINT, data={"data": query}, headers=headers, timeout=35)
            if response.status_code == 429:
                time.sleep(1.6 * (2 ** attempt))
                continue
            response.raise_for_status()
            payload = response.json()
            seen: set[str] = set()
            rows: List[Dict[str, Any]] = []
            for element in payload.get("elements", []):
                tags = element.get("tags") or {}
                name = tags.get("name")
                if not name:
                    continue
                if "lat" in element and "lon" in element:
                    lat, lon = element["lat"], element["lon"]
                else:
                    center = element.get("center") or {}
                    lat, lon = center.get("lat"), center.get("lon")
                if lat is None or lon is None:
                    continue
                poi_id = f"osm_{element['type']}_{element['id']}"
                if poi_id in seen:
                    continue
                seen.add(poi_id)
                distance_km = haversine_km(geo["lat"], geo["lon"], float(lat), float(lon))
                rows.append(
                    {
                        "poi_id": poi_id,
                        "name": name,
                        "category": infer_category(tags),
                        "lat": float(lat),
                        "lon": float(lon),
                        "distance_km": round(distance_km, 2),
                        "url": tags.get("website") or tags.get("url") or "",
                    }
                )
            rows.sort(key=lambda x: (x["distance_km"], x["name"]))
            return {
                "city_key": geo["name"].strip().lower(),
                "display_name": geo["name"],
                "center": {"lat": geo["lat"], "lon": geo["lon"]},
                "pois": rows[: max(1, min(limit, 120))],
                "error": "",
            }
        except Exception as exc:
            failure = str(exc)
            time.sleep(1.2 * (2 ** attempt))
    return {
        "city_key": geo["name"].strip().lower(),
        "display_name": geo["name"],
        "center": {"lat": geo["lat"], "lon": geo["lon"]},
        "pois": [],
        "error": failure or "Unknown Overpass error.",
    }


# ------------------------------------------------------------
# Wikivoyage retrieval
# ------------------------------------------------------------
def wikimedia_headers(contact: str) -> Dict[str, str]:
    return {"User-Agent": make_user_agent(contact), "Accept": "application/json"}


@st.cache_data(ttl=60 * 60 * 24 * 7)
def wikivoyage_title(city: str, contact: str) -> Optional[str]:
    params = {"action": "query", "list": "search", "srsearch": city, "srlimit": 1, "format": "json"}
    response = requests.get(WIKIVOYAGE_ENDPOINT, params=params, headers=wikimedia_headers(contact), timeout=15)
    if response.status_code == 403:
        return None
    response.raise_for_status()
    hits = response.json().get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0]["title"]


@st.cache_data(ttl=60 * 60 * 24 * 7)
def wikivoyage_plain_text(title: str, contact: str) -> str:
    params = {"action": "parse", "page": title, "prop": "text", "format": "json"}
    response = requests.get(WIKIVOYAGE_ENDPOINT, params=params, headers=wikimedia_headers(contact), timeout=20)
    if response.status_code == 403:
        return ""
    response.raise_for_status()
    html = response.json().get("parse", {}).get("text", {}).get("*", "")
    text = re.sub(r"<(script|style).*?>.*?</\\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<br\\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\\s*>", "\n\n", text, flags=re.I)
    text = re.sub(r"<.*?>", " ", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def split_into_chunks(text: str, target_chars: int = 900, min_chars: int = 240) -> List[str]:
    pieces = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""
    for piece in pieces:
        if len(buffer) + len(piece) + 2 <= target_chars:
            buffer = (buffer + "\n\n" + piece).strip()
        else:
            if len(buffer) >= min_chars:
                chunks.append(buffer)
            buffer = piece
    if len(buffer) >= min_chars:
        chunks.append(buffer)
    return chunks



def prepare_guide_index(city: str, contact: str) -> Dict[str, Any]:
    cache = st.session_state.setdefault("_guide_cache", {})
    if city in cache:
        return cache[city]
    title = wikivoyage_title(city, contact)
    if not title:
        cache[city] = {"title": None, "chunks": [], "matrix": None, "vectorizer": None}
        return cache[city]
    plain = wikivoyage_plain_text(title, contact)
    chunks = split_into_chunks(plain) if plain else []
    if not chunks:
        cache[city] = {"title": title, "chunks": [], "matrix": None, "vectorizer": None}
        return cache[city]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=25000)
    matrix = vectorizer.fit_transform(chunks)
    cache[city] = {"title": title, "chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}
    return cache[city]



def retrieve_guide_snippets(city: str, question: str, top_k: int, contact: str, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {"city": city, "hits": [], "note": "Guide retrieval disabled."}
    idx = prepare_guide_index(city, contact)
    if not idx.get("title") or idx.get("matrix") is None or idx.get("vectorizer") is None:
        return {"city": city, "hits": [], "note": "No guide material available."}
    q = idx["vectorizer"].transform([question])
    sims = cosine_similarity(q, idx["matrix"]).ravel()
    order = np.argsort(-sims)[: max(1, min(top_k, 8))]
    hits: List[Dict[str, Any]] = []
    for i in order:
        hits.append(
            {
                "chunk_id": f"{idx['title']}::{int(i)}",
                "source": idx["title"],
                "score": float(sims[int(i)]),
                "text": idx["chunks"][int(i)],
            }
        )
    return {"city": city, "hits": hits, "note": "ok"}


# ------------------------------------------------------------
# Tool facade
# ------------------------------------------------------------
def rank_pois_for_query(pois: List[Dict[str, Any]], query: str, city_key: str) -> List[Dict[str, Any]]:
    boosts = vote_adjustments(city_key)
    q = (query or "").strip().lower()
    ranked: List[Tuple[Tuple[float, float, str], Dict[str, Any]]] = []
    for item in pois:
        match = 1.0 if q and q in item["name"].lower() else 0.0
        category_hint = 0.6 if q and q in item.get("category", "").lower() else 0.0
        score = match + category_hint + boosts.get(item["poi_id"], 0.0) - 0.03 * item.get("distance_km", 0)
        ranked.append(((score, -item.get("distance_km", 0), item["name"]), item))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked]



def tool_find_pois(city: str, interests: List[str], radius_km: float, limit: int, query: str, contact: str) -> Dict[str, Any]:
    raw = discover_pois(city, tuple(interests), radius_km, limit, contact)
    ranked = rank_pois_for_query(raw.get("pois", []), query, raw.get("city_key", city.strip().lower()))
    return {
        "city_key": raw.get("city_key", city.strip().lower()),
        "display_name": raw.get("display_name", city),
        "center": raw.get("center", {}),
        "pois": ranked[: max(1, min(limit, 60))],
        "error": raw.get("error", ""),
    }



def tool_get_guidance(city: str, query: str, top_k: int, contact: str, enabled: bool) -> Dict[str, Any]:
    return retrieve_guide_snippets(city, query, top_k, contact, enabled)


TOOLS = [
    {
        "type": "function",
        "name": "find_pois",
        "description": "Search OpenStreetMap POIs for a destination city and return poi_id values that must be used in the itinerary.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "interests": {"type": "array", "items": {"type": "string"}},
                "radius_km": {"type": "number", "minimum": 1, "maximum": 50},
                "limit": {"type": "integer", "minimum": 1, "maximum": 60},
                "query": {"type": "string"},
            },
            "required": ["city", "interests", "radius_km", "limit", "query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_destination_guide",
        "description": "Retrieve relevant Wikivoyage text chunks for grounding. Returns chunk_id, source, score, text.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["city", "query", "top_k"],
            "additionalProperties": False,
        },
    },
]


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("The model did not return a JSON object.")
    return json.loads(text[start:end + 1])



def invalid_poi_ids(itinerary: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> List[str]:
    valid = set(catalog)
    bad: List[str] = []
    for day in itinerary.get("days", []):
        for block in BLOCK_NAMES:
            for item in day.get(block, []) or []:
                pid = item.get("poi_id")
                if pid and pid not in valid:
                    bad.append(pid)
    return sorted(set(bad))



def preserve_other_days(old: Dict[str, Any], new: Dict[str, Any], target_day: int) -> Tuple[bool, List[int]]:
    old_by_day = {int(d["day"]): d for d in old.get("days", []) if d.get("day") is not None}
    new_by_day = {int(d["day"]): d for d in new.get("days", []) if d.get("day") is not None}
    changed: List[int] = []
    for day_num, old_day in old_by_day.items():
        if day_num == target_day:
            continue
        if json.dumps(old_day, sort_keys=True) != json.dumps(new_by_day.get(day_num), sort_keys=True):
            changed.append(day_num)
    return len(changed) == 0, changed


# ------------------------------------------------------------
# OpenAI agent
# ------------------------------------------------------------
def openai_client_from_session() -> Optional[Any]:
    key = (st.session_state.get("openai_key") or "").strip()
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)



def get_item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)



def dispatch_tool(name: str, arguments: Dict[str, Any], runtime: Dict[str, Any], contact: str, rag_enabled: bool) -> str:
    started = time.time()
    trace("tool", name=name, arguments=arguments)
    if name == "find_pois":
        result = tool_find_pois(
            city=arguments["city"],
            interests=arguments["interests"],
            radius_km=float(arguments["radius_km"]),
            limit=int(arguments["limit"]),
            query=arguments["query"],
            contact=contact,
        )
        for poi in result.get("pois", []):
            runtime.setdefault("poi_catalog", {})[poi["poi_id"]] = poi
        runtime["city_key"] = result.get("city_key", runtime.get("city_key", ""))
        runtime["display_name"] = result.get("display_name", runtime.get("display_name", ""))
        runtime["map_center"] = result.get("center", runtime.get("map_center", {}))
    elif name == "get_destination_guide":
        result = tool_get_guidance(
            city=arguments["city"],
            query=arguments["query"],
            top_k=int(arguments["top_k"]),
            contact=contact,
            enabled=rag_enabled,
        )
        for hit in result.get("hits", []):
            runtime.setdefault("guide_hits", {})[hit["chunk_id"]] = hit
    else:
        result = {"error": f"Unknown tool: {name}"}
    trace("tool_result", name=name, elapsed=round(time.time() - started, 3))
    return json.dumps(result, ensure_ascii=False)



def run_openai_planner(prompt: str, model: str, contact: str, rag_enabled: bool, max_steps: int) -> Tuple[str, Dict[str, Any]]:
    client = openai_client_from_session()
    if client is None:
        raise RuntimeError("OpenAI key missing or openai package unavailable.")

    runtime: Dict[str, Any] = {"poi_catalog": {}, "guide_hits": {}, "map_center": {}}
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]

    for step in range(1, max_steps + 1):
        trace("model", step=step)
        response = client.responses.create(model=model, input=messages, tools=TOOLS, store=False)
        messages += response.output
        tool_calls = [item for item in response.output if get_item_value(item, "type") == "function_call"]
        if not tool_calls:
            return response.output_text, runtime
        for call in tool_calls:
            args = json.loads(get_item_value(call, "arguments") or "{}")
            tool_output = dispatch_tool(get_item_value(call, "name", ""), args, runtime, contact, rag_enabled)
            messages.append({
                "type": "function_call_output",
                "call_id": get_item_value(call, "call_id"),
                "output": tool_output,
            })
    raise RuntimeError("Agent stopped after reaching the maximum number of tool steps.")


# ------------------------------------------------------------
# Free fallback planner
# ------------------------------------------------------------
def bucket_for_poi(poi: Dict[str, Any]) -> str:
    cat = (poi.get("category") or "").lower()
    if "cafe" in cat or "restaurant" in cat or "bar" in cat or "pub" in cat or "nightclub" in cat:
        return "evening"
    if "park" in cat or "viewpoint" in cat or "museum" in cat or "gallery" in cat or "attraction" in cat or "historic" in cat:
        return "afternoon"
    return "morning"



def score_for_poi(poi: Dict[str, Any], interests: List[str], constraints: str, notes: str, city_key: str) -> float:
    text = f"{poi.get('name','')} {poi.get('category','')}".lower()
    boosts = vote_adjustments(city_key)
    score = 0.0
    for interest in interests:
        if interest == "food" and any(word in text for word in ["restaurant", "food_court"]):
            score += 1.2
        elif interest == "coffee" and "cafe" in text:
            score += 1.1
        elif interest == "museums" and any(word in text for word in ["museum", "gallery"]):
            score += 1.3
        elif interest == "history" and "historic" in text:
            score += 1.2
        elif interest == "outdoors" and any(word in text for word in ["park", "garden", "nature_reserve", "peak", "beach"]):
            score += 1.3
        elif interest == "scenic" and any(word in text for word in ["viewpoint", "peak", "beach"]):
            score += 1.2
        elif interest == "nightlife" and any(word in text for word in ["bar", "pub", "nightclub"]):
            score += 1.3
        elif interest == "art" and any(word in text for word in ["artwork", "gallery", "museum"]):
            score += 1.1
    if "no early" in constraints.lower() or "late start" in constraints.lower():
        if bucket_for_poi(poi) == "morning":
            score -= 0.15
    score -= poi.get("distance_km", 0) * 0.04
    score += boosts.get(poi["poi_id"], 0.0)
    if "iconic" in notes.lower() and any(word in text for word in ["museum", "attraction", "viewpoint", "historic"]):
        score += 0.4
    if "hidden gem" in notes.lower() and poi.get("distance_km", 0) > 1.2:
        score += 0.2
    return score



def concise_reason(poi: Dict[str, Any], block: str) -> str:
    cat = poi.get("category", "place")
    if block == "morning":
        return f"A gentle start built around a nearby {cat}."
    if block == "afternoon":
        return f"A stronger daytime anchor with clear sightseeing value as a {cat}."
    return f"A comfortable finish to the day around this {cat}."



def fallback_plan(city: str, days: int, pace: str, interests: List[str], constraints: str, notes: str, radius_km: float, rag_enabled: bool, contact: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trace("note", message="Using deterministic fallback planner.")
    primary = tool_find_pois(city, interests, radius_km, 45, "", contact)
    guide = tool_get_guidance(city, f"best neighborhoods, logistics, highlights for {city}", 4, contact, rag_enabled)
    catalog = {row["poi_id"]: row for row in primary.get("pois", [])}
    city_key = primary.get("city_key", city.strip().lower())

    ranked = sorted(
        primary.get("pois", []),
        key=lambda poi: score_for_poi(poi, interests, constraints, notes, city_key),
        reverse=True,
    )

    quotas = FALLBACK_THEMES.get(pace, FALLBACK_THEMES["balanced"])
    used: set[str] = set()
    day_rows: List[Dict[str, Any]] = []

    by_block: Dict[str, List[Dict[str, Any]]] = {block: [] for block in BLOCK_NAMES}
    for poi in ranked:
        by_block[bucket_for_poi(poi)].append(poi)
    for block in BLOCK_NAMES:
        if len(by_block[block]) < days:
            fill = [poi for poi in ranked if poi not in by_block[block]]
            by_block[block].extend(fill)

    for day_index in range(1, days + 1):
        row: Dict[str, Any] = {"day": day_index, "notes": "", "sources": []}
        for block in BLOCK_NAMES:
            picks: List[Dict[str, Any]] = []
            for poi in by_block[block]:
                if poi["poi_id"] in used:
                    continue
                picks.append(poi)
                used.add(poi["poi_id"])
                if len(picks) >= quotas[block]:
                    break
            if not picks:
                fallback_candidates = [poi for poi in ranked if poi["poi_id"] not in used]
                if fallback_candidates:
                    picks = [fallback_candidates[0]]
                    used.add(fallback_candidates[0]["poi_id"])
            row[block] = [{"poi_id": poi["poi_id"], "why": concise_reason(poi, block)} for poi in picks]
        row["notes"] = "Balanced automatically from live POIs, distance, interests, and saved feedback."
        row["sources"] = [{"chunk_id": hit["chunk_id"], "source": hit["source"]} for hit in guide.get("hits", [])[:3]]
        day_rows.append(row)

    itinerary = {
        "title": f"{days}-Day Plan for {primary.get('display_name', city)}",
        "city": primary.get("display_name", city),
        "days": day_rows,
        "planner_mode": "fallback",
    }
    runtime = {
        "poi_catalog": catalog,
        "map_center": primary.get("center", {}),
        "city_key": city_key,
    }
    return itinerary, runtime


# ------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------
def generation_prompt(city: str, days: int, pace: str, interests: List[str], constraints: str, notes: str, radius_km: float, use_rag: bool, fast_mode: bool) -> str:
    if fast_mode:
        instructions = f"""
Tool strategy:
- Start by calling find_pois exactly once with city={city!r}, interests={interests}, radius_km={radius_km}, limit=40, query="".
- Call find_pois a second time only if you truly need alternatives for a missing slot.
- If guide retrieval is enabled, call get_destination_guide once with a query that helps itinerary design.
"""
    else:
        instructions = f"""
Tool strategy:
- Call find_pois at least twice with different query intents when useful.
- If guide retrieval is enabled, call get_destination_guide once.
- Stay within 5 total tool calls unless absolutely necessary.
"""
    return f"""
You are an itinerary planning agent.

Create a {days}-day itinerary for {city}.
Pace: {pace}
Interests: {interests}
Constraints: {constraints}
Extra notes: {notes}
Guide retrieval enabled: {use_rag}

Hard rules:
1. You may only use poi_id values returned by find_pois.
2. Return JSON only, with no markdown and no commentary.
3. Each block should contain 1 to 2 items at most.
4. Each item must have a concise reason.
5. If no guide content is available, use an empty sources list.

{instructions}

JSON schema:
{{
  "title": "string",
  "city": "string",
  "days": [
    {{
      "day": 1,
      "morning": [{{"poi_id": "string", "why": "string"}}],
      "afternoon": [{{"poi_id": "string", "why": "string"}}],
      "evening": [{{"poi_id": "string", "why": "string"}}],
      "notes": "string",
      "sources": [{{"chunk_id": "string", "source": "string"}}]
    }}
  ]
}}
"""


# ------------------------------------------------------------
# Itinerary rendering
# ------------------------------------------------------------
def poi_label(poi_id: str, catalog: Dict[str, Dict[str, Any]]) -> str:
    item = catalog.get(poi_id, {})
    name = item.get("name", poi_id)
    category = item.get("category", "")
    return f"{name} ({category})" if category else name



def render_itinerary(itinerary: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> None:
    st.subheader(itinerary.get("title", "Itinerary"))
    st.caption(itinerary.get("city", ""))
    mode = itinerary.get("planner_mode")
    if mode == "fallback":
        st.info("This itinerary was produced by the built-in free planner because no OpenAI key was used.")
    for day in itinerary.get("days", []) or []:
        st.markdown(f"### Day {day.get('day')}")
        cols = st.columns(3)
        for i, block in enumerate(BLOCK_NAMES):
            with cols[i]:
                st.markdown(f"**{block.title()}**")
                entries = day.get(block, []) or []
                if not entries:
                    st.caption("No items")
                for entry in entries:
                    st.markdown(f"- **{poi_label(entry.get('poi_id', ''), catalog)}**  \n  {entry.get('why', '')}")
        if day.get("notes"):
            st.caption(day["notes"])
        if day.get("sources"):
            with st.expander("Guide sources", expanded=False):
                for src in day["sources"]:
                    st.markdown(f"- `{src.get('chunk_id', '')}` from **{src.get('source', '')}**")


# ------------------------------------------------------------
# Map rendering
# ------------------------------------------------------------
def extract_map_points(itinerary: Dict[str, Any], catalog: Dict[str, Dict[str, Any]], day_filter: Optional[int]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for day in itinerary.get("days", []) or []:
        day_num = int(day.get("day", -1))
        if day_filter is not None and day_num != day_filter:
            continue
        for block in BLOCK_NAMES:
            for entry in day.get(block, []) or []:
                poi = catalog.get(entry.get("poi_id", ""))
                if not poi:
                    continue
                points.append(
                    {
                        "day": day_num,
                        "block": block,
                        "name": poi.get("name", entry.get("poi_id", "")),
                        "category": poi.get("category", ""),
                        "poi_id": poi.get("poi_id", ""),
                        "lat": float(poi["lat"]),
                        "lon": float(poi["lon"]),
                    }
                )
    return points



def extract_paths(itinerary: Dict[str, Any], catalog: Dict[str, Dict[str, Any]], day_filter: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for day in itinerary.get("days", []) or []:
        day_num = int(day.get("day", -1))
        if day_filter is not None and day_num != day_filter:
            continue
        coords: List[List[float]] = []
        for block in BLOCK_NAMES:
            entries = day.get(block, []) or []
            if not entries:
                continue
            poi = catalog.get(entries[0].get("poi_id", ""))
            if poi and poi.get("lon") is not None and poi.get("lat") is not None:
                coords.append([float(poi["lon"]), float(poi["lat"])])
        if len(coords) >= 2:
            rows.append({"day": day_num, "path": coords})
    return rows



def estimate_zoom(points: List[Dict[str, Any]]) -> int:
    lat_span = max(p["lat"] for p in points) - min(p["lat"] for p in points)
    lon_span = max(p["lon"] for p in points) - min(p["lon"] for p in points)
    span = max(lat_span, lon_span)
    if span < 0.01:
        return 14
    if span < 0.03:
        return 13
    if span < 0.08:
        return 12
    if span < 0.18:
        return 11
    if span < 0.35:
        return 10
    return 9



def render_map(itinerary: Dict[str, Any], catalog: Dict[str, Dict[str, Any]], center: Dict[str, Optional[float]], dark: bool, key: str) -> None:
    day_options = ["All"] + [int(day.get("day")) for day in itinerary.get("days", []) if day.get("day") is not None]
    if key in st.session_state and st.session_state[key] not in day_options:
        del st.session_state[key]
    choice = st.selectbox("Map day filter", options=day_options, index=0, key=key)
    day_filter = None if choice == "All" else int(choice)

    points = extract_map_points(itinerary, catalog, day_filter)
    paths = extract_paths(itinerary, catalog, day_filter)
    if not points:
        st.info("No map points available yet.")
        return

    latitude = center.get("lat")
    longitude = center.get("lon")
    if latitude is None or longitude is None:
        latitude = float(np.mean([p["lat"] for p in points]))
        longitude = float(np.mean([p["lon"] for p in points]))

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_radius=38,
        radius_min_pixels=3,
        radius_max_pixels=9,
        get_fill_color=[30, 136, 229, 190],
        get_line_color=[255, 255, 255, 230],
        line_width_min_pixels=1,
        pickable=True,
    )
    layers = [scatter]
    if paths:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=paths,
                get_path="path",
                get_color=[235, 87, 87, 175],
                width_min_pixels=2,
                width_max_pixels=5,
            )
        )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=estimate_zoom(points), pitch=0),
        map_style=DARK_MAP if dark else LIGHT_MAP,
        tooltip={"text": "{name}\nDay {day} • {block}\n{category}\n{poi_id}"},
    )
    st.pydeck_chart(deck, use_container_width=True)


# ------------------------------------------------------------
# Planning orchestration
# ------------------------------------------------------------
def generate_itinerary(city: str, days: int, pace: str, interests: List[str], constraints: str, notes: str, radius_km: float, model: str, contact: str, rag_enabled: bool, fast_mode: bool, max_steps: int) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    prompt = generation_prompt(city, days, pace, interests, constraints, notes, radius_km, rag_enabled, fast_mode)
    client = openai_client_from_session()
    if client is None:
        itinerary, runtime = fallback_plan(city, days, pace, interests, constraints, notes, radius_km, rag_enabled, contact)
        return itinerary, runtime, json.dumps(itinerary, ensure_ascii=False)

    raw_text, runtime = run_openai_planner(prompt, model, contact, rag_enabled, max_steps)
    itinerary = extract_json_object(raw_text)
    return itinerary, runtime, raw_text



def apply_generation_result(itinerary: Dict[str, Any], runtime: Dict[str, Any], raw_text: str) -> None:
    catalog = dict(runtime.get("poi_catalog", {}))
    bad = invalid_poi_ids(itinerary, catalog)
    if bad:
        raise ValueError(f"Unknown poi_id values detected: {bad}")
    if not catalog:
        raise ValueError("No POIs were returned, so the itinerary cannot be trusted.")
    st.session_state["itinerary"] = itinerary
    st.session_state["poi_catalog"] = catalog
    st.session_state["map_center"] = runtime.get("map_center", {})
    st.session_state["city_key"] = runtime.get("city_key", itinerary.get("city", "").strip().lower())
    st.session_state["last_raw_output"] = raw_text
    save_snapshot()


# ------------------------------------------------------------
# Refine helpers
# ------------------------------------------------------------
def refine_with_fallback(existing: Dict[str, Any], catalog: Dict[str, Dict[str, Any]], request: str) -> Dict[str, Any]:
    updated = json.loads(json.dumps(existing))
    request_l = request.lower()
    if "outdoors" in request_l:
        scenic = [p for p in catalog.values() if any(k in p.get("category", "").lower() for k in ["park", "viewpoint", "peak", "garden", "nature_reserve"])]
        scenic.sort(key=lambda p: p.get("distance_km", 999))
        for day in updated.get("days", []):
            if scenic:
                day["afternoon"] = [{"poi_id": scenic[0]["poi_id"], "why": "Shifted toward a more outdoor-focused anchor."}]
    if "chill" in request_l or "relax" in request_l:
        for day in updated.get("days", []):
            if day.get("evening"):
                day["notes"] = ((day.get("notes") or "") + " Evening kept deliberately light.").strip()
    return updated


# ------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")
restore_snapshot()

st.title("Voyager Agent: Trip Planner Capstone")
st.caption("Live OpenStreetMap data, optional Wikivoyage grounding, OpenAI tool calling when available, and a free fallback planner when it is not.")

with st.sidebar:
    st.header("Settings")

    st.text_input(
        "OpenAI API key",
        type="password",
        key="openai_key",
        help="Optional. Leave blank to use the free fallback planner."
    )

    # Model dropdown (replaces text_input)
    MODEL_OPTIONS = [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
    ]

    default_index = MODEL_OPTIONS.index(DEFAULT_OPENAI_MODEL) if DEFAULT_OPENAI_MODEL in MODEL_OPTIONS else 0

    st.selectbox(
        "Model",
        options=MODEL_OPTIONS,
        index=default_index,
        key="model_name",
    )

    st.checkbox("Fast mode", value=True, key="fast_mode")
    st.checkbox("Enable Wikivoyage retrieval", value=False, key="rag_enabled")
    st.checkbox("Show execution trace", value=True, key="show_trace")
    st.checkbox("Autosave itinerary locally", value=True, key="autosave_enabled")
    st.checkbox("Dark map", value=False, key="dark_map")

    st.text_input(
        "User-Agent contact",
        value="your-email@example.com",
        key="contact_email"
    )

    st.slider(
        "Max tool steps",
        min_value=3,
        max_value=10,
        value=5,
        key="max_steps"
    )

plan_tab, refine_tab, feedback_tab = st.tabs(["Plan", "Refine", "Feedback"])

with plan_tab:
    left, right = st.columns(2)
    with left:
        city = st.text_input("Destination city", value="Lisbon, Portugal")
        days = st.slider("Trip length", min_value=1, max_value=7, value=3)
        pace = st.selectbox("Pace", ["relaxed", "balanced", "packed"], index=1)
        radius_km = st.slider("POI search radius (km)", min_value=1, max_value=30, value=8)
    with right:
        interests = st.multiselect("Interests", VALID_INTERESTS, default=["food", "history", "scenic"])
        constraints = st.text_area("Constraints", value="No early mornings. Keep walking moderate.")
        notes = st.text_area("Extra notes", value="Include one iconic highlight and one more local-feeling stop.")

    if st.button("Generate itinerary", type="primary"):
        reset_trace()
        try:
            itinerary, runtime, raw_text = generate_itinerary(
                city=city,
                days=days,
                pace=pace,
                interests=interests,
                constraints=constraints,
                notes=notes,
                radius_km=radius_km,
                model=st.session_state.get("model_name", DEFAULT_OPENAI_MODEL),
                contact=st.session_state.get("contact_email", "your-email@example.com"),
                rag_enabled=st.session_state.get("rag_enabled", False),
                fast_mode=st.session_state.get("fast_mode", True),
                max_steps=st.session_state.get("max_steps", 5),
            )
            apply_generation_result(itinerary, runtime, raw_text)
            st.success("Itinerary generated and saved.")
        except Exception as exc:
            trace("error", where="generate", message=str(exc))
            st.error(f"Planning failed: {exc}")

    current_itinerary = st.session_state.get("itinerary")
    current_catalog = st.session_state.get("poi_catalog", {})
    current_center = st.session_state.get("map_center", {})
    if current_itinerary and current_catalog:
        st.divider()
        render_itinerary(current_itinerary, current_catalog)
        st.subheader("Map")
        render_map(current_itinerary, current_catalog, current_center, st.session_state.get("dark_map", False), key="map_day_plan")
        with st.expander("Raw itinerary JSON", expanded=False):
            st.json(current_itinerary)
        st.download_button(
            "Download itinerary.json",
            data=json.dumps(current_itinerary, ensure_ascii=False, indent=2),
            file_name="itinerary.json",
            mime="application/json",
        )
    else:
        st.info("Generate a plan to see the itinerary and map.")

    if st.session_state.get("show_trace", True):
        render_trace()

with refine_tab:
    itinerary = st.session_state.get("itinerary")
    catalog = st.session_state.get("poi_catalog", {})
    center = st.session_state.get("map_center", {})
    if not itinerary or not catalog:
        st.info("Create an itinerary first.")
    else:
        render_itinerary(itinerary, catalog)
        st.subheader("Map")
        render_map(itinerary, catalog, center, st.session_state.get("dark_map", False), key="map_day_refine")

        st.divider()
        global_request = st.text_input("Refine whole plan", value="Make the plan a little more outdoorsy and keep evenings calm.")
        if st.button("Apply refinement"):
            if openai_client_from_session() is None:
                updated = refine_with_fallback(itinerary, catalog, global_request)
                st.session_state["itinerary"] = updated
                save_snapshot()
                st.success("Fallback refinement applied.")
            else:
                reset_trace()
                prompt = f"""
Edit this existing itinerary JSON. Keep the same schema. Use only poi_id values you already know or fetch via tools. Output JSON only.
Refinement request: {global_request}
Existing JSON: {json.dumps(itinerary, ensure_ascii=False)}
"""
                try:
                    raw_text, runtime = run_openai_planner(
                        prompt,
                        st.session_state.get("model_name", DEFAULT_OPENAI_MODEL),
                        st.session_state.get("contact_email", "your-email@example.com"),
                        st.session_state.get("rag_enabled", False),
                        st.session_state.get("max_steps", 5),
                    )
                    updated = extract_json_object(raw_text)
                    merged_catalog = dict(catalog)
                    merged_catalog.update(runtime.get("poi_catalog", {}))
                    bad = invalid_poi_ids(updated, merged_catalog)
                    if bad:
                        raise ValueError(f"Refined plan referenced unknown poi_id values: {bad}")
                    st.session_state["itinerary"] = updated
                    st.session_state["poi_catalog"] = merged_catalog
                    st.session_state["map_center"] = runtime.get("map_center", center)
                    save_snapshot()
                    st.success("Refinement applied.")
                except Exception as exc:
                    trace("error", where="refine", message=str(exc))
                    st.error(f"Refinement failed: {exc}")

        day_numbers = [int(day.get("day")) for day in itinerary.get("days", []) if day.get("day") is not None]
        if day_numbers:
            selected_day = st.selectbox("Regenerate just one day", options=day_numbers)
            single_day_request = st.text_area("Request for that day", value="Swap the afternoon for something different and keep the evening cozy.")
            if st.button("Regenerate selected day"):
                if openai_client_from_session() is None:
                    updated = json.loads(json.dumps(itinerary))
                    for day in updated.get("days", []):
                        if int(day.get("day", -1)) == selected_day and day.get("evening"):
                            day["notes"] = "Day adjusted locally without changing the rest of the plan."
                    st.session_state["itinerary"] = updated
                    save_snapshot()
                    st.success("Selected day lightly adjusted with fallback mode.")
                else:
                    reset_trace()
                    prompt = f"""
Edit the itinerary JSON below.
Only modify day {selected_day}. Every other day must remain exactly unchanged.
Use only valid poi_id values from existing or newly fetched tool outputs.
Output JSON only.
Request: {single_day_request}
Existing JSON: {json.dumps(itinerary, ensure_ascii=False)}
"""
                    try:
                        raw_text, runtime = run_openai_planner(
                            prompt,
                            st.session_state.get("model_name", DEFAULT_OPENAI_MODEL),
                            st.session_state.get("contact_email", "your-email@example.com"),
                            st.session_state.get("rag_enabled", False),
                            st.session_state.get("max_steps", 5),
                        )
                        updated = extract_json_object(raw_text)
                        merged_catalog = dict(catalog)
                        merged_catalog.update(runtime.get("poi_catalog", {}))
                        ok, changed_days = preserve_other_days(itinerary, updated, selected_day)
                        if not ok:
                            raise ValueError(f"The model changed other day numbers too: {changed_days}")
                        bad = invalid_poi_ids(updated, merged_catalog)
                        if bad:
                            raise ValueError(f"Regenerated day referenced unknown poi_id values: {bad}")
                        st.session_state["itinerary"] = updated
                        st.session_state["poi_catalog"] = merged_catalog
                        st.session_state["map_center"] = runtime.get("map_center", center)
                        save_snapshot()
                        st.success(f"Day {selected_day} updated.")
                    except Exception as exc:
                        trace("error", where="regenerate_day", message=str(exc))
                        st.error(f"Day regeneration failed: {exc}")

        if st.session_state.get("show_trace", True):
            render_trace()

with feedback_tab:
    itinerary = st.session_state.get("itinerary")
    catalog = st.session_state.get("poi_catalog", {})
    city_key = st.session_state.get("city_key", "")
    if not itinerary or not catalog:
        st.info("Generate an itinerary first.")
    else:
        referenced_ids: List[str] = []
        for day in itinerary.get("days", []) or []:
            for block in BLOCK_NAMES:
                for entry in day.get(block, []) or []:
                    if entry.get("poi_id"):
                        referenced_ids.append(entry["poi_id"])
        referenced_ids = list(dict.fromkeys(referenced_ids))
        st.caption("Votes are stored locally and used to nudge later POI ranking for the same city.")
        for poi_id in referenced_ids:
            poi = catalog.get(poi_id, {})
            cols = st.columns([5, 1, 1])
            with cols[0]:
                label = f"**{poi.get('name', poi_id)}**  \n`{poi_id}`  \n{poi.get('category', '')}"
                if poi.get("url"):
                    label += f"  \n{poi['url']}"
                st.markdown(label)
            if cols[1].button("👍", key=f"up_{poi_id}"):
                append_vote(city_key, poi_id, "up")
                st.toast(f"Upvoted {poi.get('name', poi_id)}")
            if cols[2].button("👎", key=f"down_{poi_id}"):
                append_vote(city_key, poi_id, "down")
                st.toast(f"Downvoted {poi.get('name', poi_id)}")
