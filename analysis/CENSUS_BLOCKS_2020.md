# San Francisco 2020 Census Blocks

The official U.S. Census Bureau 2020 P.L. 94-171 TIGER/Line Block, Block Group,
and Tract layers for San Francisco County (state FIPS `06`, county FIPS `075`)
are stored under `/share/data/school_choice/Census/2020/`. Original archives,
ISO metadata, and extracted Shapefile sidecars are retained there.

The Block source is:

https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/06_CALIFORNIA/06075/tl_2020_06075_tabblock20.zip

SHA-256: `0f858e6c7748070b3f1a564ec39a55c2ef6913afb41dc6adb935621c771516b9`.

The FRL analyses use each polygon's `GEOID20` to map student latitude and
longitude to the corresponding 2020 block.
Production loaders select all three layers through `geography_vintage: "2020"`.
