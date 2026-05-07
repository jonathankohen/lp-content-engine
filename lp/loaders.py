from .config import SKILL_GRAPH_DIR


def load_skill_graph() -> str:
    """Read all markdown files in content-skill-graph/ into one concatenated string."""
    parts = []
    for md_file in sorted(SKILL_GRAPH_DIR.rglob("*.md")):
        rel = md_file.relative_to(SKILL_GRAPH_DIR.parent)
        parts.append(f"## {rel}\n\n{md_file.read_text().strip()}")
    return "\n\n---\n\n".join(parts)


def load_artist_mappings() -> dict[str, str]:
    """Parse artists.md markdown table → {tribute_name: original_artist}."""
    path = SKILL_GRAPH_DIR / "engine" / "artists.md"
    mappings: dict[str, str] = {}
    if not path.exists():
        return mappings
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if (
            not stripped.startswith("|")
            or "---" in stripped
            or "Tribute Act" in stripped
        ):
            continue
        cols = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cols) >= 2 and cols[0]:
            mappings[cols[0]] = cols[1]
    return mappings
