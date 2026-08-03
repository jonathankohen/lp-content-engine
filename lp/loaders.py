import re

from .config import SKILL_GRAPH_DIR


def load_skill_graph() -> str:
    """Read all markdown files in content-skill-graph/ into one concatenated string."""
    parts = []
    for md_file in sorted(SKILL_GRAPH_DIR.rglob("*.md")):
        rel = md_file.relative_to(SKILL_GRAPH_DIR.parent)
        parts.append(f"## {rel}\n\n{md_file.read_text().strip()}")
    return "\n\n---\n\n".join(parts)


def _normalize_act(name: str) -> str:
    """Fold an act name to a form that survives the ways it is written down.

    ``artists.md`` is hand-maintained and Airtable is the source of act names,
    so the two drift: "Arrival from Sweden" against "Arrival From Sweden",
    "Concert of Kings" against "Elvis: The Concert of Kings", "The Dolly Show"
    against Airtable's filing-order "Dolly Show, The". Every one of those was a
    silent lookup miss, and a miss means no original artist, which costs the act
    its trivia and historical-fact posts as well as its hashtags.
    """
    n = (name or "").strip().lower()
    n = re.sub(r"^(?:the)\s+", "", n)
    n = re.sub(r",\s*the$", "", n)
    # Drop a leading or trailing descriptor: "elvis: the concert of kings",
    # "kiss the sky: a jimi hendrix tribute".
    if ":" in n:
        head, tail = (p.strip() for p in n.split(":", 1))
        n = max((head, tail), key=len)
    n = re.sub(r"^(?:the)\s+", "", n)
    return re.sub(r"[^a-z0-9]+", "", n)


class ArtistMappings(dict):
    """``{tribute_act: original_artist}`` whose lookups tolerate name drift.

    ``get()`` tries the exact key first, so an explicit entry always wins, then
    falls back to the normalized form. Behaves like a plain dict everywhere
    else, so no call site had to change.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._normalized = {}
        for key, value in self.items():
            self._normalized.setdefault(_normalize_act(key), value)

    def get(self, key, default=""):  # noqa: A003, matches dict.get
        if key in self:
            return self[key]
        return self._normalized.get(_normalize_act(key), default)


def load_artist_mappings() -> ArtistMappings:
    """Parse artists.md markdown table → {tribute_name: original_artist}."""
    path = SKILL_GRAPH_DIR / "engine" / "artists.md"
    mappings: dict[str, str] = {}
    if not path.exists():
        return ArtistMappings()
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
    return ArtistMappings(mappings)
