"""Script to find Objaverse UIDs for target objects (cereal box, cup/mug)."""

from __future__ import annotations

import objaverse


def search_objects(
    query: str, annotations: dict, top_k: int = 10
) -> list[tuple[str, dict]]:
    """Search annotations for objects matching query in
    name/tags/description."""
    query_lower = query.lower()
    results = []
    for uid, meta in annotations.items():
        name = (meta.get("name") or "").lower()
        tags = " ".join(t.get("name", "") for t in (meta.get("tags") or [])).lower()
        desc = (meta.get("description") or "").lower()
        if query_lower in name or query_lower in tags or query_lower in desc:
            results.append((uid, meta))
        if len(results) >= top_k:
            break
    return results


def print_results(query: str, results: list[tuple[str, dict]]) -> None:
    """Print search results in readable format."""
    print(f"\n=== '{query}' — {len(results)} results ===")
    for uid, meta in results:
        name = meta.get("name", "<no name>")
        tags = [t.get("name", "") for t in (meta.get("tags") or [])][:5]
        dl_count = meta.get("downloadCount", 0)
        like_count = meta.get("likeCount", 0)
        print(f"  UID: {uid}")
        print(f"    name: {name}")
        print(f"    tags: {tags}")
        print(f"    downloads: {dl_count}  likes: {like_count}")


def test_load_objaverse_objects() -> None:
    """Test loading Objaverse annotations and searching for target objects."""
    print("Loading Objaverse annotations...")
    annotations = objaverse.load_annotations()
    print(f"Total objects: {len(annotations)}")

    for query in ["cereal box", "cereal", "milk", "milk carton", "cup", "mug"]:
        results = search_objects(query, annotations, top_k=10)
        print_results(query, results)
