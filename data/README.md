# Input data contract

Two Clarivate InCites Benchmarking & Analytics exports (Excel), saved with these names in the folder passed to `--data-dir`:

| File | InCites entity | Rows in the study export |
|---|---|---|
| `Incites_Locations.xlsx` | Locations (countries/regions) | 181 |
| `Incites_Organizations.xlsx` | Organizations | 10,741 |

Filters used for both exports: Web of Science category *Computer Science, Artificial Intelligence*; publication years 2015–2025; document types Article and Review; exported on 9 October 2026. Record the InCites dataset update date shown in your session when re-exporting, because citation counts change over time.

Required columns (exact InCites headers): Name; Web of Science Documents; Times Cited; Category Normalized Citation Impact; Collab-CNCI; % International Documents; % Domestic Single Documents; % Domestic Multiple Documents; % International Bilateral Documents; % International Trilateral Documents; % International Quadrilateral + Documents. Other columns are ignored.

Processing applied in code: England, Scotland, Wales and Northern Ireland are dropped from the location file (the United Kingdom aggregate is kept); units with fewer than 100 documents are excluded (sensitivity: 50, 200, 250, 500).

Hierarchy map (`--edges`) — a curated parent–child organization map in CSV with columns: child (exact InCites name, or `re:` regular expression applied to eligible names), parent, relation_type, basis (manual or reviewed name rule), curated_on, evidence_url, verification. The map used in the study is included as `hierarchy_edges.csv` (137 rows: manual edges and manually reviewed name-pattern rules, which expand to 189 edges among the eligible organizations). It was curated by the author from organization names, public institutional sources and naming conventions, and contains no InCites indicator values. `synthetic_hierarchy_edges.csv` is a small example in the same format that matches the names produced by `scripts/make_synthetic_data.py`.
