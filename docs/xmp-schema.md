# photo-tools XMP / IPTC schema

This document describes the XMP/IPTC metadata photo-tools reads and
writes, the taxonomy it uses for keyword tags, and the conventions
consumers can rely on.

---

## 1. Output fields

photo-tools writes exactly **two keyword/tag fields** plus a small custom
XMP namespace for tool-private metadata.

### 1.1 Keyword/tag fields

| Field | Content | Separator | Notes |
| --- | --- | --- | --- |
| `dc:subject` (mirrored to `IPTC:Keywords`) | Leaf names only | flat list | MWG-aligned; what Lightroom and most tools actually read |
| `digiKam:TagsList` | Full hierarchy paths | `/` | digiKam's native hierarchy field |

Both fields use **Titlecase** for every path segment (see §3).

For example:

```
dc:subject       = [Chiara, Johannes, Italy, Lazio, Rome, Municipio Roma I,
                    Colosseum, Balustrade, Rail, Building, Cityscape]
IPTC:Keywords    = (same as dc:subject)
digiKam:TagsList = [
    People/Chiara, People/Johannes,                  # written by digiKam, not us
    Places/Italy/Lazio/Rome/Municipio Roma I,
    Landmarks/Colosseum,
    Objects/Balustrade, Objects/Rail,
    Scenes/Building, Scenes/Cityscape,
]
```

### 1.2 photo-tools custom namespace

Namespace prefix: **`photo-tools`** (also surfaces as `XMP-phototools` via the
exiftool config).

Namespace URI: **`https://github.com/j23n/photo-tools/ns/1.0/`**

| Field | Type | Purpose |
| --- | --- | --- |
| `photo-tools:TaggerVersion` | string (`"YYYY.N"`) | Sentinel — presence means file was tagged by this tool. A mismatched value triggers re-tag on next run. |
| `photo-tools:TaggedAt` | ISO 8601 timestamp | When the file was last tagged. |
| `photo-tools:CountryCode` | ISO 3166-1 alpha-2 (uppercase, e.g. `IT`) | Country code from reverse geocoding. Kept out of the keyword space. |
| `photo-tools:CLIPEmbedding` | base64 of float32 vector | Cached image embedding for similarity search. |
| `photo-tools:CLIPModel` | string (e.g. `ViT-B-32/laion2b_s34b_b79k`) | Model identifier for the cached embedding. |

### 1.3 Fields photo-tools does not write

These fields are deliberately absent from output:

- `XMP-lr:HierarchicalSubject` — redundant with `digiKam:TagsList`.
- `MicrosoftPhoto:LastKeywordXMP`
- `MediaPro:CatalogSets`
- `MicrosoftPhoto:CategorySet`

### 1.4 Sentinel mechanics

- A file is considered "already tagged" iff `photo-tools:TaggerVersion` is
  present.
- If the stored version differs from the current `TaggerVersion`, the file is
  re-tagged automatically.
- `--force` re-tags regardless. `--clear-all` wipes everything first.

---

## 2. Taxonomy

Top-level roots in `digiKam:TagsList`:

```
People/<name>                                        ← digiKam-owned
Places/<Country>[/<Region>[/<City>[/<Neighborhood>]]]
Landmarks/<name>
Objects/<TopLevel>/<SubCategory>/<Leaf>
Scenes/<TopLevel>/<SubCategory>/<Leaf>
Text/<phrase>
```

Objects and Scenes use 2–3 path segments (leaf nodes may be 2 levels when
the subcategory itself is the useful term, e.g. `Nature/Forest`).

### 2.1 People

`People/*` is **owned by digiKam face recognition**. photo-tools never writes
or removes anything under this root.

### 2.2 Places

A single nested path from reverse geocoding. Missing levels collapse — there
are no placeholders.

```
Places/Italy/Lazio/Rome/Municipio Roma I    ← all four available
Places/Italy/Rome                            ← region missing, collapse
Places/France                                ← only country known
```

The country code is written separately to `photo-tools:CountryCode` (see
§1.2) and is not part of the keyword space.

### 2.3 Objects

Concrete things detected by RAM++. Configured in `taxonomy.py`:

- `max_tags`: 5

Tags use the following top-level hierarchy (22 roots):

| Root | Covers |
| --- | --- |
| `Animal` | Mammals, birds, fish, insects, reptiles, amphibians, invertebrates, aquatic life |
| `Appliance` | Home appliances (standalone subcategory under `Household`) |
| `Art` | Paintings, drawings, sculptures, murals, calligraphy, collage |
| `Artifact` | Books, historical/ceremonial objects, fossils, relics |
| `Clothing` | Tops, bottoms, outerwear, footwear, headwear, accessories, swimwear, workwear |
| `Container` | Bags, boxes, bottles, jars, baskets, cases |
| `Electronics` | Computers, phones, audio/video gear, cameras, networking, displays |
| `Food` | Fruit, vegetables, meat, seafood, dairy, baked goods, beverages, condiments |
| `Furniture` | Seating, beds, tables, storage, office furniture |
| `Household` | Kitchenware, linens, lighting, timekeeping, bathroom, baby items |
| `Instrument` | String, wind, brass, percussion, keyboard instruments |
| `Medical` | Medicine, equipment, mobility aids, safety |
| `Nature` | Rocks, minerals, wood, water, sky, celestial, natural materials |
| `Person` | Roles, costumes, uniforms, professions depicted in-image |
| `Plant` | Trees, flowers, shrubs, succulents, ferns, vines, fungi, crops, herbs |
| `Sport` | Equipment and gear by sport |
| `Structure` | Buildings, bridges, towers, infrastructure, architectural elements |
| `Tool` | Hand tools, power tools, gardening, woodworking, measurement |
| `Toy` | Dolls, games, puzzles, toy vehicles, play equipment |
| `Urban` | Signs, street elements, lighting, windows, fences, walls |
| `Vehicle` | Cars, trucks, buses, aircraft, watercraft, rail, bikes |
| `Weapon` | Firearms, blades, projectiles, armor, military equipment |

### 2.4 Scenes

Scene/setting classification from RAM++. Configured in `taxonomy.py`:

- `max_tags`: 3

Tags are first gated against RAM++'s per-class thresholds scaled by
`THRESHOLD_MARGIN` (default 1.10) — a prediction is kept only if its
sigmoid score is at least `threshold * THRESHOLD_MARGIN`. Per-category
`max_tags` is then applied to the survivors in score-descending order.

Scenes use the following top-level hierarchy (6 roots):

| Root | Subcategories |
| --- | --- |
| `Interior` | Domestic, Work, Other, Vehicle |
| `Nature` | Coastal, Forest, Landscape, Mountain, Phenomenon, Vegetation, Water |
| `Sky` | (flat — celestial bodies and events) |
| `Urban` | District, Park, Square, Street, Waterfront |
| `Venue` | Agricultural, Civic, Cultural, Dining, Historical, Hospitality, Industrial, Market, Religious, Residential, Retail, Sport, Transport, Wellness |
| `Weather` | (flat — season and condition tags) |

### 2.5 Landmarks

Named landmarks identified by CLIP embedding similarity against a curated
index, gated by GPS proximity. A single flat segment — the landmark name is
Titlecased and emitted as the leaf with no geographic nesting:

```
Landmarks/Colosseum
Landmarks/Eiffel Tower
```

Landmarks are independent of `Places/`. A photo of the Colosseum gets both
the reverse-geocoded Places path *and* the Landmarks tag; the former answers
"where was this taken" and the latter answers "what is this". Landmark
lookup requires GPS (from EXIF or an explicit fallback) and a built index;
without either it is skipped silently.

### 2.6 Text

Visible text detected by PaddleOCR. Each accepted phrase becomes a single
flat segment:

```
Text/Pizza
Text/Caffè Centrale
```

OCR also writes IPTC `ImageRegion` entries for each detected phrase so
consumers can locate the text in the image. Confidence and length filters
are configured under `ocr.*` in `default_config.yaml`.

---

## 3. Casing rules

Every segment of every keyword/tag path is **Titlecase**, including
geocoder-derived values.

- ASCII text, single-word: `str.title()` is sufficient (`rome` → `Rome`).
- Multi-word, mixed-language, Roman numerals: use the
  [`titlecase`](https://pypi.org/project/titlecase/) PyPI library
  (`municipio roma i` → `Municipio Roma I`, `città del vaticano` → `Città del
  Vaticano`).

The `titlecase` library handles small words ("of", "the", "del"), Roman
numerals, and acronyms out of the box.

---

## 4. Interop notes

- **MWG / Lightroom / digiKam** all read leaf names from `dc:subject`. Keeping
  it leaf-only matches the convention.
- **digiKam** writes `People/*` itself (face recognition) and reads/writes
  `digiKam:TagsList` for hierarchy. Round-trip is clean.
- **Lightroom** writes hierarchy to `lr:HierarchicalSubject` (`|` separator).
  photo-tools does not mirror to it; users who want Lightroom-side hierarchy
  can run exiftool to copy `digiKam:TagsList` → `lr:HierarchicalSubject`
  themselves.
- **photo-tools custom namespace** is registered in
  `src/photo_tools/exiftool_phototools.config`. exiftool needs `-config
  exiftool_phototools.config` to read/write the namespace fields by name; the
  raw XMP is always preserved regardless.

---

## 5. Versioning

`TaggerVersion` follows `YYYY.N` (e.g. `2026.2`). Bump `N` when:

- Any field in §1 is added, removed, or repurposed.
- Taxonomy roots in §2 change shape.
- Casing or normalization rules in §3 change.

A version bump triggers automatic re-tagging on next `photo-tools tag` run
because the stored `TaggerVersion` no longer matches the running one.
