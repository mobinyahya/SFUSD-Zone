# Generate Zones from Pickle Files

This document explains how to use zone dictionaries from pickle files to run school assignment simulations.

## Overview

The `Zone_Generation` pipeline produces pickle files containing dictionaries that map each zone ID to a list of geounits (census block groups, blocks, or attendance areas). To use these zones in the `student-assignment` simulation:

1. **Convert** the pickle file to CSV format
2. **Register** the zone file in the path configuration
3. **Create** a policy configuration referencing the zone
4. **Run** the simulation

## Step-by-Step Guide

### Step 1: Convert Pickle to CSV

```bash
cd student-assignment

# Convert a single pickle file
python scripts/generators/generate_zone_from_pickle.py \
    --input /path/to/Generated_Zones/Zones_6/FRL_Dev_0.20/cost_123.456.pkl \
    --output /share/data/school_choice/Data/assignment/zones/6zone-frl20-1.csv \
    --building-blocks block_group

# Or convert all pickle files in a directory
python scripts/generators/generate_zone_from_pickle.py \
    --input-dir /path/to/Generated_Zones/Zones_6/FRL_Dev_0.20/ \
    --output-dir /share/data/school_choice/Data/assignment/zones/ \
    --building-blocks block_group
```

### Step 2: Register the Zone File

Add the zone to the executable run config's scenario overrides:

```yaml
data:
  scenario: legacy
  overrides:
    sources:
      assignment.zones:
        custom-6zone:
          path: Data/assignment/zones/6zone-frl20-1.csv
          root: data
          classification: public
```

### Step 3: Create/Update Policy Config

Either use the template at `configs/policy_configs/custom_zones+reserves.yaml` or create your own:

```yaml
# configs/policy_configs/my_custom_zones.yaml
assignment-algorithm: DA
ctip-options:
- 1

restrict-zone: true
guard-rails: 1

reserve-settings:
  column: median_hh_income
  thresholds: [95292]
  lower_disadvantaged: true
  citywide_only: false
  reserve_fraction: [0.57, 0.43]

policies:
- custom-6zone  # Must match the key in assignment.zones

zone-building-blocks: block_group  # Must match how zones were generated
designate: true
ties-options:
- MTB
```

### Step 4: Update Base Config

Ensure your `configs/base_config.yaml` references your policy config:

```yaml
subconfigs:
  - my_custom_zones  # Points to my_custom_zones.yaml in policy_configs/
```

### Step 5: Run the Simulation

```bash
cd /path/to/student-assignment
uv run python run_custom_config.py --config-path <your-config>.yaml
```

## Zone File Format

The CSV format expected by the simulation:
- **One row per zone**
- **Comma-separated geounit IDs** in each row
- No header row

Example (3 zones with block groups):
```csv
60750101001,60750101002,60750102003
60750201001,60750201002
60750301001,60750301002,60750301003
```

## Building Blocks

The `zone-building-blocks` setting must match how your zones were generated:

| Building Block | Description | Geounit IDs |
|----------------|-------------|-------------|
| `block_group` | Census block groups | 12-digit FIPS codes |
| `block` | Census blocks | 15-digit FIPS codes |
| `attendance_area` | School attendance areas | 3-digit school IDs |

## Troubleshooting

### "Zone file not found"
- Ensure the direct source path under `assignment.zones` is correct.
- Check that the zone key in `policies` matches the scenario map key.

### "Geounit not found in zone"
- Ensure `zone-building-blocks` matches how the pickle was generated
- Verify that student data uses the same geounit type (block_group vs block)

### Validation
Check your pickle file structure:
```python
import pickle
with open("path/to/zone.pkl", "rb") as f:
    zone_dict = pickle.load(f)
print(f"Number of zones: {len(zone_dict)}")
for zone_id, geounits in zone_dict.items():
    print(f"Zone {zone_id}: {len(geounits)} geounits")
```
