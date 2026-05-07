# Content Scoring

This system scores and ranks candidate topics before post generation. Only the top-scoring topics (one per remaining day in the current week, max 7) get generated and scheduled. Scoring happens before generation to avoid paying for content that won't be used.

Show announcements from the Airtable calendar are exempt — they always get drafted regardless of score.

## Criteria

Each topic is scored on five criteria. Scores sum to a maximum of 1.0.

---

### Relevance — 0.30

How directly related is this story to Love Productions or an act LP represents?

- News about an LP tribute act itself: 1.0
- News about the original artist a tribute act is based on: 0.8
- General music industry news with no LP connection: 0.3 or lower
- News about tribute bands LP does not represent: 0.0 — discard entirely

---

### Freshness — 0.30

How recent is the story? The scale is linear:

- Today: 1.0
- ~6 months ago: 0.5
- ~1 year ago or older: 0.0

Estimate from the publication date or event date in the summary.

---

### Velocity — 0.10

The current volume of conversation around this story — how many sources picked it up.

- Measured by: number of search results returned, and how widely distributed they are across outlets
- Social media sources (Twitter/X threads, TikTok, Reddit discussions) score higher than a single press release
- No results or a single source: 0.1 or lower

---

### Virality — 0.10

The *potential acceleration* of this story — how likely it is to spread further.

- Distinct from velocity: a story can have low velocity (not yet widely covered) but high virality potential (emotionally resonant, tied to a trending moment)
- Indicators: celebrity angle, nostalgia hook, surprising fact, cultural anniversary, recent controversy
- Estimate from the headline and summary — this is a qualitative judgment

---

### Uniqueness — 0.20

Is this a genuinely interesting angle, or generic filler?

- Compare against the **Found News spreadsheet** (the same dedup source used by `lp/sheets.py`) to confirm this hasn't been covered before
- Trivia, behind-the-scenes facts, anniversary milestones, and unexpected connections score highest
- Routine tour announcements or generic "artist releases album" stories score lower unless the artist is highly relevant

---

## Exclusivity Boost — +1.0

Consult [engine/artists.md](artists.md) for the full tribute act roster. Artists tagged **Top of Roster** or **Exclusive** in Airtable receive a +1.0 bonus added to their base score, making their effective maximum 2.0. This guarantees exclusive acts always rank above non-exclusive acts with equal or better content.

Two exceptions that count as exclusive even without the Airtable tag:
- **Tony Danza**
- **The Rocket Man Show**

---

## Minimum Threshold

Topics with a base score below **0.40** (before the exclusivity boost) should not be posted, even if they would otherwise fill an available slot. An empty slot is better than mediocre content.
