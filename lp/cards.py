"""
Branded image cards, rendered locally with Pillow. No AI call, no external service.

The client asked for "hard numbers with a graphic" and shorter posts (direction
2026-07-31). Text-only drafts cannot answer that, so this module renders the
card types that carry proof visually:

* :func:`render_quote_card`, published praise (the buyer and press pull-quotes
  harvested from act pages by ``lp.scrape.extract_page_quotes``).
* :func:`render_stat_card`, a single hard number with its label.
* :func:`render_tour_poster`, an act's upcoming dates under its own key art.
  This is the only tour-date renderer: it absorbed an earlier
  ``render_tour_card`` that set the same rows over a darkened act photo. Both
  drew tour dates, so they were reconciled into one rather than left to drift.

Quote and stat cards are drawn over the act's own photo where one is available
(darkened, so text stays legible) and over a flat dark ground otherwise. Tour
posters instead give the artwork its own undarkened panel, since key art is
dense and usually carries the act's name already. Everything is deterministic:
same inputs render the same file, so a re-run does not churn Buffer assets.

Rendering is local; Buffer needs a public URL for an image asset, so the caller
uploads the finished PNG (see ``lp.wordpress.upload_media``) and passes that URL
to ``post_draft_to_buffer``.
"""

import hashlib
import logging
import os
import re
import textwrap
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .artist_links import display_act
from .scrape import _TIMEOUT, _UA

log = logging.getLogger(__name__)

# Brand palette. RED, PAPER and BODY are the computed values read off
# loveproductions.com itself: background #FFFFFF, body text rgb(62,62,62),
# headings rgb(255,0,0). The site is a light theme, so tour posters follow it.
RED = (255, 0, 0)
CYAN = (5, 172, 223)
INK = (17, 17, 17)
PAPER = (255, 255, 255)
MUTED = (176, 176, 176)

# Light-theme tokens, used by the tour poster.
BODY = (62, 62, 62)         # site body copy
SUBTLE = (130, 130, 130)    # venue names, the third rank of information
HAIRLINE = (226, 226, 226)  # row separators

# Card geometry. LinkedIn's feed crops to roughly 1.91:1; Instagram and Facebook
# take the square well, so two sizes cover all three channels. "poster" is
# Instagram's tallest allowed ratio (4:5), which buys the extra rows a tour date
# list needs without the feed cropping it.
SIZES = {
    "linkedin": (1200, 627),
    "square": (1080, 1080),
    "poster": (1080, 1350),
}

# Official act key art, pulled from each act's own loveproductions.com page.
# Client-owned, so it is the right image to build a tour poster on.
KEY_ART_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "act-key-art"
)

# Typeface. loveproductions.com sets everything in **Paralucent**, served from
# Adobe Fonts (use.typekit.net). That licence covers web embedding only, so the
# webfont files are deliberately not downloaded and converted for use here.
#
# To render posters in the real face, install Paralucent as a desktop font
# (Adobe Fonts desktop sync with the client's Creative Cloud account, or a
# desktop licence from Device Fonts), then drop the files in ~/Library/Fonts or
# point LP_FONT_DIR at them. They are picked up automatically below, and the
# posters change with no code edit.
#
# Until then the fallback is Avenir Next, the closest geometric humanist sans
# on macOS: same open apertures and near-circular bowls. Arial and DejaVu sit
# behind it for Linux and CI, where output will look noticeably more generic.
_FONT_DIRS = [
    os.environ.get("LP_FONT_DIR", ""),
    os.path.expanduser("~/Library/Fonts"),
    "/Library/Fonts",
]

# Paralucent file-stem prefixes to prefer per weight, best first. The site uses
# weights 500, 600 and 700, so those are the ones worth matching.
_PARALUCENT_STEMS = {
    "black":   ("paralucentheavy", "paralucentblack", "paralucentbold", "paralucentdemibold"),
    "bold":    ("paralucentdemibold", "paralucentsemibold", "paralucentbold", "paralucentmedium"),
    "regular": ("paralucentmedium", "paralucentbook", "paralucentregular", "paralucentlight"),
}

# Fallbacks in preference order. An entry is a path, or (path, face index) for
# a .ttc collection: Avenir Next ships as one file holding every weight.
_FONT_CANDIDATES = {
    "bold": (
        ("/System/Library/Fonts/Avenir Next.ttc", 2),      # Demi Bold, ~600
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ),
    "black": (
        ("/System/Library/Fonts/Avenir Next.ttc", 8),      # Heavy, ~800
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    "regular": (
        ("/System/Library/Fonts/Avenir Next.ttc", 7),      # Regular
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ),
}

_EXTS = (".otf", ".ttf", ".ttc")


def _find_paralucent(weight: str) -> str | None:
    """Path to an installed Paralucent file for ``weight``, or None."""
    stems: dict[str, str] = {}
    for directory in _FONT_DIRS:
        if not directory or not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in _EXTS:
                continue
            key = "".join(c for c in stem.lower() if c.isalnum())
            stems.setdefault(key, os.path.join(directory, name))

    for prefix in _PARALUCENT_STEMS.get(weight, ()):
        for key, path in stems.items():
            if key.startswith(prefix):
                return path
    return None


def _font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    """Load the best available font for ``weight`` at ``size``.

    Real Paralucent wins if it is installed, then the fallback chain.
    """
    real = _find_paralucent(weight)
    candidates = ((real,) if real else ()) + _FONT_CANDIDATES.get(weight, ())
    for entry in candidates:
        path, index = entry if isinstance(entry, tuple) else (entry, 0)
        if not path or not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    log.warning("No TrueType font found for weight %r, falling back to bitmap default", weight)
    return ImageFont.load_default(size)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return int(draw.textbbox((0, 0), text, font=font)[2])


def _fit_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    weight: str,
    max_width: int,
    max_height: int,
    start_size: int,
    min_size: int = 22,
) -> tuple[list[str], ImageFont.FreeTypeFont, int]:
    """Wrap ``text`` and shrink the type until the block fits the given box.

    Returns the wrapped lines, the chosen font and the line height. Long quotes
    and short ones therefore both fill the card sensibly instead of overflowing
    or floating in space.
    """
    for size in range(start_size, min_size - 1, -2):
        font = _font(weight, size)
        # Estimate characters per line from average glyph width, then wrap.
        avg = max(_text_width(draw, "abcdefghijklmnopqrstuvwxyz", font) / 26, 1)
        lines = textwrap.wrap(text, width=max(int(max_width / avg), 12))
        line_h = int(size * 1.34)
        if not lines:
            continue
        too_wide = any(_text_width(draw, ln, font) > max_width for ln in lines)
        if not too_wide and len(lines) * line_h <= max_height:
            return lines, font, line_h
    font = _font(weight, min_size)
    line_h = int(min_size * 1.34)
    return textwrap.wrap(text, width=40)[:8], font, line_h


def _background(size: tuple[int, int], image_url: str | None) -> Image.Image:
    """Return the card ground: the act's darkened photo, or flat brand ink."""
    canvas = Image.new("RGB", size, INK)
    if not image_url:
        return canvas
    try:
        resp = requests.get(image_url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
        if resp.status_code != 200:
            return canvas
        photo = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001, a missing photo is a graceful miss
        log.info("Card background fetch failed for %s: %s", image_url, exc)
        return canvas

    # Cover-fit: scale to fill, centre-crop the overflow.
    scale = max(size[0] / photo.width, size[1] / photo.height)
    photo = photo.resize((max(int(photo.width * scale), size[0]),
                          max(int(photo.height * scale), size[1])), Image.LANCZOS)
    left = (photo.width - size[0]) // 2
    top = (photo.height - size[1]) // 2
    photo = photo.crop((left, top, left + size[0], top + size[1]))

    # Darken and soften so white type stays readable over any photo.
    photo = ImageEnhance.Brightness(photo).enhance(0.30)
    photo = photo.filter(ImageFilter.GaussianBlur(3))

    # Not every act page offers a clean performance shot; several use a busy
    # tour poster or logo, which competes with the type no matter how far the
    # brightness drops. A left-to-right scrim holds the text side down while
    # letting the right side of the image stay visible.
    scrim = Image.new("L", (size[0], 1))
    for x in range(size[0]):
        ratio = x / max(size[0] - 1, 1)
        scrim.putpixel((x, 0), int(232 * (1 - ratio) ** 1.5))
    scrim = scrim.resize(size, Image.BILINEAR)
    photo = Image.composite(Image.new("RGB", size, INK), photo, scrim)
    return photo


def display_name(act: str) -> str:
    """Act name as a reader expects it: "Platters, The" becomes "The Platters".

    Airtable files several acts surname-first so they sort tidily in a list. Set
    at poster scale that reads as a database export rather than a tour poster.
    """
    name = (act or "").strip()
    if name.lower().endswith(", the"):
        return "The " + name[: -len(", the")].strip()
    return name


def _slug(text: str) -> str:
    """Lowercase, hyphenated key for matching act names to key-art filenames."""
    out = "".join(c if c.isalnum() else "-" for c in (text or "").lower())
    return "-".join(p for p in out.split("-") if p)


def key_art_for(act: str, art_dir: str = KEY_ART_DIR) -> str | None:
    """Path to an act's key-art file, or None.

    Files are named by a slug of the act name, but Airtable and the key-art
    pull do not always agree on the exact wording ("Dolly Show, The" against
    "the-dolly-show"), so an exact slug match is tried first and a containment
    match second. Matching on the longest candidate avoids "reza" claiming a
    file belonging to a longer name that merely contains it.
    """
    if not act or not os.path.isdir(art_dir):
        return None
    want = _slug(act)
    files = [f for f in os.listdir(art_dir) if not f.startswith(".")
             and os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
    stems = {os.path.splitext(f)[0]: os.path.join(art_dir, f) for f in files}

    if want in stems:
        return stems[want]
    hits = [p for stem, p in stems.items() if stem in want or want in stem]
    return max(hits, key=len) if hits else None


def _open_image(src: str | None) -> Image.Image | None:
    """Open ``src`` as RGB, whether it is a local path or an http(s) URL.

    Key art is on disk; the fallback act photo is a scraped URL. One loader
    keeps the poster renderer indifferent to which it was handed.
    """
    if not src:
        return None
    try:
        if src.startswith(("http://", "https://")):
            resp = requests.get(src, timeout=_TIMEOUT, headers={"User-Agent": _UA})
            if resp.status_code != 200:
                return None
            return Image.open(BytesIO(resp.content)).convert("RGB")
        return Image.open(src).convert("RGB")
    except Exception as exc:  # noqa: BLE001, a bad image should not kill a run
        log.info("Card art unreadable at %s: %s", src, exc)
        return None


def _fit_art(src: str, box: tuple[int, int], light: bool = False) -> Image.Image | None:
    """Fit the key art into ``box`` whole, over a blurred fill of itself.

    The art is *contained*, not cover-cropped, and left at full brightness: on a
    tour poster the act's own artwork is the point, not a backdrop for type.
    Nearly all the key art is square while this box is landscape, so a cover
    crop cut the top off the very thing that identifies the act (it beheaded the
    "BOHEMIAN QUEEN" logo). The leftover width is filled with a blurred, dimmed
    copy of the same image, which reads as intentional where flat black read as
    a mistake.
    """
    art = _open_image(src)
    if art is None:
        return None

    if light:
        # Plain white surround, matching the site's own clean white pages. A
        # blurred wash of the artwork was tried here and read as muddy colour
        # bleed against white; on a light page the letterbox is the tidier answer.
        panel = Image.new("RGB", box, PAPER)
    else:
        cover = max(box[0] / art.width, box[1] / art.height)
        back = art.resize((max(int(art.width * cover), box[0]),
                           max(int(art.height * cover), box[1])), Image.LANCZOS)
        left = (back.width - box[0]) // 2
        top = (back.height - box[1]) // 2
        panel = back.crop((left, top, left + box[0], top + box[1]))
        panel = panel.filter(ImageFilter.GaussianBlur(28))
        panel = ImageEnhance.Brightness(panel).enhance(0.45)

    contain = min(box[0] / art.width, box[1] / art.height)
    front = art.resize((max(int(art.width * contain), 1),
                        max(int(art.height * contain), 1)), Image.LANCZOS)
    panel.paste(front, ((box[0] - front.width) // 2, (box[1] - front.height) // 2))
    return panel


def _footer_top(size: tuple[int, int], pad: int) -> int:
    """Y coordinate where the act-name footer begins; content must stay above it."""
    return size[1] - pad - 52


def _finish(img: Image.Image, act: str, size: tuple[int, int],
            light: bool = False) -> Image.Image:
    """Draw the footer rule, act name and agency mark shared by every card."""
    act = display_act(act)
    draw = ImageDraw.Draw(img)
    pad = int(size[0] * 0.066)
    draw.rectangle([0, size[1] - 10, size[0], size[1]], fill=RED)

    act_font = _font("bold", 27 if size[0] > 1100 else 30)
    draw.text((pad, size[1] - pad - 34), act.upper(), font=act_font,
              fill=BODY if light else PAPER)

    mark_font = _font("regular", 21 if size[0] > 1100 else 23)
    mark = "LOVEPRODUCTIONS.COM"
    draw.text((size[0] - pad - _text_width(draw, mark, mark_font), size[1] - pad - 30),
              mark, font=mark_font, fill=SUBTLE if light else MUTED)
    return img


def _out_path(kind: str, seed: str, out_dir: str) -> str:
    """Deterministic filename, so the same card never renders twice."""
    os.makedirs(out_dir, exist_ok=True)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return os.path.join(out_dir, f"lp_{kind}_{digest}.png")


def render_quote_card(
    quote: str,
    attribution: str,
    act: str,
    *,
    out_dir: str,
    size: str = "linkedin",
    background_url: str | None = None,
) -> str | None:
    """Render a published-praise card. Returns the PNG path, or None on failure.

    The quote is the subject of the card, so it is set large and the agency
    furniture stays small. Quotes are trimmed from the end (never reworded) when
    they are too long to set legibly.
    """
    quote = " ".join((quote or "").split()).strip('"“”')
    act = (act or "").strip()
    if len(quote) < 20 or not act:
        return None
    if len(quote) > 260:
        cut = quote[:260].rsplit(" ", 1)[0]
        quote = cut + "..."

    dims = SIZES.get(size, SIZES["linkedin"])
    img = _background(dims, background_url)
    draw = ImageDraw.Draw(img)
    pad = int(dims[0] * 0.066)
    inner = dims[0] - pad * 2

    # Opening mark, drawn as a graphic element rather than punctuation.
    mark_font = _font("black", int(dims[1] * 0.20))
    draw.text((pad - 6, pad - int(dims[1] * 0.06)), "“", font=mark_font, fill=RED)

    # Reserve the footer strip up front so a long quote shrinks to fit rather
    # than running into the act name.
    attr_font = _font("regular", 26 if dims[0] > 1100 else 29)
    attribution = " ".join((attribution or "").split())
    attr_lines = textwrap.wrap(attribution, width=54)[:2] if attribution else []
    attr_block = len(attr_lines) * int(attr_font.size * 1.3)

    top = pad + int(dims[1] * 0.13)
    body_box = _footer_top(dims, pad) - top - attr_block - int(dims[1] * 0.05)
    lines, font, line_h = _fit_lines(
        draw, quote, "bold", inner, body_box, start_size=int(dims[1] * 0.072)
    )
    for i, line in enumerate(lines):
        draw.text((pad, top + i * line_h), line, font=font, fill=PAPER)

    attr_y = top + len(lines) * line_h + int(dims[1] * 0.045)
    for line in attr_lines:
        draw.text((pad, attr_y), line, font=attr_font, fill=CYAN)
        attr_y += int(attr_font.size * 1.3)

    img = _finish(img, act, dims)
    path = _out_path(f"quote_{size}", f"{act}|{quote}|{attribution}|{size}|{background_url}", out_dir)
    img.save(path, "PNG", optimize=True)
    log.info("Rendered quote card: %s", path)
    return path


def render_tour_poster(
    act: str,
    dates: list[dict],
    *,
    out_dir: str,
    size: str = "poster",
    art_path: str | None = None,
    background_url: str | None = None,
    max_rows: int = 12,
    min_row_px: int = 34,
    art_ratio: float | None = None,
    eyebrow: str = "TOUR DATES",
    light: bool = True,
    show_venue: bool = True,
) -> str | None:
    """Render a tour poster: the act's own key art above its upcoming dates.

    The acts do not publish real tour posters. What they post is single-show
    flyers, usually venue-branded and expired within the week (see
    ``assets/act-key-art/TOUR-POSTERS-FINDINGS.md``), so the multi-date poster
    has to be built rather than found.

    This is the single tour-date renderer. It replaced an earlier
    ``render_tour_card`` that set the same rows *over* a darkened act photo:
    that layout fought the artwork, since key art is dense and usually carries
    the act's own name and logo, so type laid over it was unreadable whichever
    way the brightness went. Here the art keeps its own panel at full brightness
    and the dates get a solid ground beneath it.

    Art is resolved in order: an explicit ``art_path``, then the act's official
    key art on disk, then ``background_url`` (a scraped act photo). Key art
    wins because it is client-owned and made for the act; the scrape is only a
    fallback for acts with no file. With none of the three the dates fill the
    whole card rather than the card failing.

    ``dates`` are the dicts from ``lp.sheets.upcoming_tour_dates``, soonest
    first. Rows are capped so the list stays legible: an act with thirty dates
    gets the soonest ones and a "+N more" line, because a poster crammed to the
    footer sells nothing. Squares suit an Instagram carousel, where every slide
    must share one aspect ratio; ``poster`` (4:5) is the tallest Instagram
    leaves uncropped and fits the most dates.

    ``size="linkedin"`` works but is the weakest of the three: 1200x627 leaves
    the art a shallow strip and room for only about six rows. Nothing asks for
    it today, since the carousel sends LinkedIn a square slide as well. Give the
    landscape format a side-by-side layout (art left, dates right) before
    shipping it, rather than this stacked one.

    ``light`` follows loveproductions.com: white page, rgb(62,62,62) text, red
    headings and accents, all read off the live site. Pass ``light=False`` for
    the old dark treatment.

    ``show_venue`` adds the venue as a third column after the city. Venue data
    is complete in the sheet today, so this is normally on, and it forces a
    single-column list because three fields need the full width.
    """
    act = (act or "").strip()
    rows = [
        d for d in (dates or [])
        if d.get("date") and (d.get("city") or d.get("venue"))
    ]
    if not act or not rows:
        return None

    dims = SIZES.get(size, SIZES["poster"])
    pad = int(dims[0] * 0.066)
    inner = dims[0] - pad * 2

    art_src = art_path or key_art_for(act) or background_url
    # A landscape card has little height to spare, so the art takes a smaller
    # share of it or there is no room left for the dates that are the point.
    # Adding venue names costs half the rows (three fields need one full-width
    # column), so the art gives some height back to keep the list worth reading.
    if art_ratio is None:
        if dims[0] > dims[1]:
            art_ratio = 0.40
        else:
            art_ratio = 0.46 if (show_venue and any(r.get("venue") for r in rows)) else 0.52
    ground = PAPER if light else INK
    text_hi = BODY if light else PAPER
    text_lo = SUBTLE if light else MUTED

    art_h = int(dims[1] * art_ratio) if art_src else 0
    img = Image.new("RGB", dims, ground)
    art = _fit_art(art_src, (dims[0], art_h), light=light) if art_src else None
    if art is None:
        art_h = 0
    else:
        img.paste(art, (0, 0))
        # Fade the foot of the art into the panel. A hard seam reads as two
        # images stuck together rather than one poster.
        fade_h = int(art_h * 0.22)
        mask = Image.new("L", (1, fade_h))
        for i in range(fade_h):
            mask.putpixel((0, i), int(255 * (i / max(fade_h - 1, 1)) ** 1.4))
        mask = mask.resize((dims[0], fade_h), Image.BILINEAR)
        img.paste(Image.new("RGB", (dims[0], fade_h), ground), (0, art_h - fade_h), mask)

    draw = ImageDraw.Draw(img)
    y = art_h + int(dims[1] * 0.030) if art_h else pad

    eyebrow_font = _font("bold", int(dims[1] * 0.024))
    draw.text((pad, y), eyebrow.upper(), font=eyebrow_font, fill=RED)
    y += int(eyebrow_font.size * 1.8)

    # The key art almost always carries the act name already, so the panel
    # repeats it small rather than at poster scale, which would say it twice.
    # Held to a single line by giving it barely more than one line's height:
    # the long names ("A1A: The Original Jimmy Buffett Tribute") wrapped to two
    # and cost three tour dates apiece, which is a bad trade for a name the
    # artwork above already states.
    name_lines, name_font, name_h = _fit_lines(
        draw, display_name(act).upper(), "black", inner, int(dims[1] * 0.056),
        start_size=int(dims[1] * 0.040), min_size=19,
    )
    for line in name_lines:
        draw.text((pad, y), line, font=name_font, fill=text_hi)
        y += name_h

    y += int(dims[1] * 0.012)
    draw.rectangle([pad, y, pad + 96, y + 7], fill=RED)
    y += int(dims[1] * 0.034)

    # Venue names are the third field, so a row needs the full width and the
    # two-column layout is off whenever they are shown. Every act in the sheet
    # currently has a venue on every date, so in practice this is the layout.
    venues = show_venue and any(r.get("venue") for r in rows)

    # Fit as many dates as the panel takes at a legible row height, then say how
    # many were left off. Shrinking type to force all thirty on would make the
    # poster unreadable, which is the failure this format exists to avoid.
    #
    # Without venues a single column of eight rows left half the panel empty
    # while the act had seventeen dates, so that case runs in two columns.
    available = _footer_top(dims, pad) - y
    per_col = max(available // min_row_px, 1)
    cols = 1 if venues else (2 if len(rows) > per_col and inner >= 640 else 1)
    gutter = int(dims[0] * 0.045)
    col_w = (inner - gutter) // 2 if cols == 2 else inner

    shown = min(len(rows), max_rows * cols, per_col * cols)
    extra = len(rows) - shown
    if extra:
        available -= int(min_row_px * 1.2)
    drawn = rows[:shown]

    rows_per_col = -(-shown // cols)  # ceiling, so column one is never shorter
    row_h = max(int(available / max(rows_per_col, 1)), min_row_px)
    row_font = _font("bold", min(int(row_h * 0.52), int(dims[1] * 0.026)))
    city_font = _font("bold", row_font.size)
    venue_font = _font("regular", int(row_font.size * 0.92))

    # A run that crosses a year boundary reads as a sorting bug without the
    # year: The Platters go "FEB 25, MAR 10, FEB 27" because those February
    # dates are two years out. The year is only added when it is needed, since
    # on a single-year run it is noise.
    multiyear = len({d["date"].year for d in drawn}) > 1
    fmt = "%b %d '%y" if multiyear else "%b %d"

    def _date_str(row: dict) -> str:
        """One date, or a range for a residency merged by collapse_residencies.

        A run of nights at one venue prints as "SEP 08-14" rather than seven
        identical rows, which is how a real tour admat handles it and which
        stops one residency hiding every other city on the poster.
        """
        start = row["date"].strftime(fmt).upper()
        end = row.get("date_end")
        if not end:
            return start
        # Same month: "SEP 08-14". Across months: "SEP 28-OCT 04".
        tail = end.strftime("%d" if end.month == row["date"].month else fmt).upper()
        return f"{start}-{tail}"

    date_strs = [_date_str(d) for d in drawn]
    date_col = max(_text_width(draw, s, row_font) for s in date_strs) + int(dims[0] * 0.030)

    def _place(row: dict) -> str:
        """City and region, or "" when there is no city.

        A region on its own is not a place: the cruise dates on Legends of
        Classic Rock leave the city blank and put the port in the venue column,
        which rendered as "AUG 02   FL   Port Canaveral". Returning "" here
        hands the row's full width to the venue, giving "AUG 02  PORT CANAVERAL".
        """
        city = (row.get("city") or "").strip()
        if not city:
            return ""
        region = (row.get("region") or "").strip()
        return f"{city}, {region}" if region else city

    # City column is sized to the longest city actually present, capped so a
    # "The Tarkington at the Center for the Performing Arts" cannot squeeze the
    # venue out entirely.
    city_col = 0
    if venues:
        widest = max((_text_width(draw, _place(r).upper(), city_font) for r in drawn), default=0)
        city_col = min(widest + int(dims[0] * 0.030), int(inner * 0.42))

    def _venue_text(row: dict) -> str:
        """The venue with any city and state it repeats stripped off.

        The sheet often stores "Newtown Theater, Newtown PA" while the city
        column beside it already says NEWTOWN, PA. Dropping the repeat is both
        better typography and the difference between fitting and not.
        """
        venue = (row.get("venue") or "").strip()
        city = (row.get("city") or "").strip()
        if not venue or not city:
            return venue
        head, sep, tail = venue.rpartition(",")
        if not sep:
            return venue
        # "Newtown PA", "Newtown, PA" and "Newtown" all count as a repeat.
        tail_words = re.sub(r"[^a-z ]", " ", tail.lower()).split()
        city_words = re.sub(r"[^a-z ]", " ", city.lower()).split()
        region = (row.get("region") or "").strip().lower()
        allowed = set(city_words) | ({region} if region else set())
        if tail_words and all(w in allowed for w in tail_words):
            return head.strip() or venue
        return venue

    # Horizontal fit. The type already shrinks to fit the number of rows; this
    # does the same for the width, so a long venue name makes the whole list a
    # little smaller instead of getting cut off. Scaling everything together
    # keeps the rows aligned, where shrinking one row's venue would not.
    def _columns(rf, cf):
        dc = max(_text_width(draw, t, rf) for t in date_strs) + int(dims[0] * 0.030)
        cc = 0
        if venues:
            widest = max((_text_width(draw, _place(r).upper(), cf) for r in drawn), default=0)
            cc = min(widest + int(dims[0] * 0.030), int(inner * 0.42))
        return dc, cc

    def _overflow(cf, vf, dc, cc):
        """Widest amount by which any row exceeds its column, 0 when all fit."""
        worst = 0
        for r in drawn:
            if venues and _place(r):
                need = _text_width(draw, _venue_text(r), vf) - (col_w - dc - cc)
            elif venues:
                need = _text_width(draw, (r.get("venue") or "").strip().upper(), cf) - (col_w - dc)
            else:
                need = _text_width(draw, _place(r).upper(), cf) - (col_w - dc)
            worst = max(worst, need)
        return worst

    # Deliberately not named `size`: that is the caller's "poster"/"square"
    # argument, and shadowing it puts the font size into the output filename,
    # which would let a poster and a square render collide on one file.
    type_px = row_font.size
    while type_px > 15 and _overflow(city_font, venue_font, date_col, city_col) > 0:
        type_px -= 1
        row_font = _font("bold", type_px)
        city_font = _font("bold", type_px)
        venue_font = _font("regular", max(int(type_px * 0.92), 1))
        date_col, city_col = _columns(row_font, city_font)

    def _clip(text: str, font, limit: int) -> str:
        """Trim to fit, on a word boundary, with an ellipsis.

        The old rule chopped two characters at a time and appended a full stop,
        which produced "Newtown Theater, Newto." and "Allen Theatre & Backstag.",
        both of which read as a rendering fault rather than an abbreviation.
        """
        if limit <= 0:
            return ""
        if _text_width(draw, text, font) <= limit:
            return text
        # The horizontal-fit loop above shrinks the type until everything fits,
        # so reaching here means even the minimum size was not enough. Logged
        # because it should be rare, and is worth knowing about when it is not.
        log.info("Tour poster: trimming over-long text %r", text[:60])
        words = text.split()
        while len(words) > 1:
            words.pop()
            candidate = " ".join(words).rstrip(",&-") + "…"
            if _text_width(draw, candidate, font) <= limit:
                return candidate
        # A single word too wide for the column still has to be cut somewhere.
        while text and _text_width(draw, text + "…", font) > limit:
            text = text[:-1]
        return text + "…" if text else ""


    top_y = y
    for i, (row, date_str) in enumerate(zip(drawn, date_strs)):
        col_x = pad + (i // rows_per_col) * (col_w + gutter)
        row_y = top_y + (i % rows_per_col) * row_h

        place = _place(row) or row.get("venue", "")
        draw.text((col_x, row_y), date_str, font=row_font, fill=RED)

        if venues:
            venue = _venue_text(row)
            # A row whose city is missing (several are cruise ports) gives its
            # whole width to the venue rather than leaving a gap.
            if not _place(row):
                draw.text((col_x + date_col, row_y),
                          _clip((row.get("venue") or "").strip().upper(),
                                city_font, col_w - date_col),
                          font=city_font, fill=text_hi)
            else:
                draw.text((col_x + date_col, row_y),
                          _clip(place.upper(), city_font, city_col - 12),
                          font=city_font, fill=text_hi)
                draw.text((col_x + date_col + city_col, row_y + 1),
                          _clip(venue, venue_font, col_w - date_col - city_col),
                          font=venue_font, fill=text_lo)
        else:
            draw.text((col_x + date_col, row_y),
                      _clip(place.upper(), city_font, col_w - date_col),
                      font=city_font, fill=text_hi)

    y = top_y + rows_per_col * row_h
    if extra:
        more_font = _font("bold", int(row_font.size * 0.86))
        draw.text((pad, y + 4), f"+ {extra} MORE DATE{'S' if extra > 1 else ''}",
                  font=more_font, fill=text_lo)

    img = _finish(img, "", dims, light=light)
    seed = f"{act}|{size}|{art_src}|{eyebrow}|{light}|{venues}|" + "|".join(
        f"{d['date']}{d.get('city','')}{d.get('region','')}{d.get('venue','')}" for d in drawn
    )
    path = _out_path(f"tourposter_{size}", seed, out_dir)
    img.save(path, "PNG", optimize=True)
    log.info("Rendered tour poster: %s (%d of %d dates)", path, shown, len(rows))
    return path


def render_stat_card(
    value: str,
    label: str,
    act: str,
    *,
    out_dir: str,
    size: str = "linkedin",
    background_url: str | None = None,
    context: str = "",
) -> str | None:
    """Render a single-number card: the figure, what it counts, and the act.

    One number per card, deliberately. The point is that a buyer takes the fact
    in without reading, which a second figure defeats.
    """
    value = " ".join((value or "").split())
    label = " ".join((label or "").split())
    act = (act or "").strip()
    if not value or not label or not act:
        return None

    dims = SIZES.get(size, SIZES["linkedin"])
    img = _background(dims, background_url)
    draw = ImageDraw.Draw(img)
    pad = int(dims[0] * 0.066)
    inner = dims[0] - pad * 2

    num_lines, num_font, _ = _fit_lines(
        draw, value, "black", inner, int(dims[1] * 0.42),
        start_size=int(dims[1] * 0.34), min_size=48,
    )
    numeral = num_lines[0]

    # Measure the whole stack before drawing any of it, then centre it in the
    # space above the footer. Drawing from a fixed top left the number stranded
    # near the middle with a band of dead space above it.
    ctx_font = _font("regular", 25 if dims[0] > 1100 else 28)
    ctx_lines = textwrap.wrap(" ".join(context.split()), width=58) if context else []
    if len(ctx_lines) > 2:
        ctx_lines = []  # too long to set whole; see the note on half sentences below
    lab_lines, lab_font, lab_h = _fit_lines(
        draw, label, "bold", inner, int(dims[1] * 0.20),
        start_size=int(dims[1] * 0.062), min_size=24,
    )
    num_h = draw.textbbox((0, 0), numeral, font=num_font)[3]
    ctx_h = len(ctx_lines) * int(ctx_font.size * 1.3) + (8 if ctx_lines else 0)
    stack = (num_h + int(dims[1] * 0.035) + 8 + int(dims[1] * 0.055)
             + len(lab_lines) * lab_h + ctx_h)

    # The context is the first thing to go if the stack will not fit: a
    # half-drawn sentence ("...five times since") is worse than none at all.
    if stack > _footer_top(dims, pad) - pad and ctx_lines:
        ctx_lines, stack = [], stack - ctx_h
    top = max(pad, (_footer_top(dims, pad) - stack) // 2)
    draw.text((pad, top), numeral, font=num_font, fill=PAPER)

    # Measure the drawn glyphs rather than deriving from the point size: a
    # numeral with a descending comma ("6,000") sits lower than the nominal
    # baseline, and the rule was cutting through it.
    y = draw.textbbox((pad, top), numeral, font=num_font)[3] + int(dims[1] * 0.035)
    draw.rectangle([pad, y, pad + 96, y + 8], fill=RED)

    y += int(dims[1] * 0.055)
    for line in lab_lines:
        draw.text((pad, y), line, font=lab_font, fill=PAPER)
        y += lab_h

    for line in ctx_lines:
        draw.text((pad, y + 8), line, font=ctx_font, fill=MUTED)
        y += int(ctx_font.size * 1.3)

    img = _finish(img, act, dims)
    path = _out_path(f"stat_{size}", f"{act}|{value}|{label}|{context}|{size}|{background_url}", out_dir)
    img.save(path, "PNG", optimize=True)
    log.info("Rendered stat card: %s", path)
    return path
