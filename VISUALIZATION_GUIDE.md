# LSCM Atlas Visualization Guide

## Overview

We provide two visualization scripts to show the automatic mesh segmentation and LSCM atlas results:

1. **`visualize_charts.py`** - Shows mesh segmentation with colored charts
2. **`visualize_atlas.py`** - Interactive atlas viewer with UV parameterizations

## Quick Start

### 1. Visualize Chart Segmentation

Shows the automatic segmentation with each chart in a different color:

```bash
source .venv/bin/activate
python visualize_charts.py
```

**What you'll see:**
- Original mesh divided into colored regions (each color = one chart)
- Statistics about chart sizes
- Distribution of small/medium/large charts

**To change mesh:** Edit `MESH_FILE` in the script (line 12)

### 2. Interactive Atlas Viewer

Shows segmentation + individual parameterized charts:

```bash
source .venv/bin/activate
python visualize_atlas.py
```

**What you'll see:**
- **Left side**: Original mesh colored by charts
- **Right side**: Selected chart with its LSCM parameterization

**UI Controls:**
- **Slider**: Select which chart to view
- **"Show 3D" checkbox**:
  - ✅ Checked: Show 3D chart with UV texture overlay
  - ❌ Unchecked: Show flattened UV representation
- Chart statistics displayed (distortion metrics, face count, etc.)

## Example Output

### Spot Mesh Results:
```
Total charts: 276
Successful LSCM: 267 (96.7%)
Mean angle distortion: 1.000-1.022 (near-perfect conformal)
```

### Bunny Mesh Results:
```
Total charts: 4,583
Successful LSCM: 4,522 (98.7%)
Mean angle distortion: 1.001-1.008 (near-perfect conformal)
```

## Configuration Options

Both scripts can be customized by editing variables at the top:

```python
# In visualize_charts.py or visualize_atlas.py
MESH_FILE = 'meshes/spot.obj'      # Change mesh
PERCENTILE = 95                     # Sharpness threshold (95 = top 5%)
MAX_CHARTS_TO_SHOW = 10            # Limit for performance (atlas viewer only)
```

### Available Meshes:
- `meshes/spot.obj` - Dog (3,225 vertices, 5,856 faces)
- `meshes/bunny.obj` - Stanford Bunny (34,834 vertices, 69,451 faces) ⚠️ Large!
- `meshes/armadillo.obj` - Armadillo (49,990 vertices, 99,976 faces) ⚠️ Very large!

**Note:** Bunny and Armadillo will take 2-5 minutes to process and may show thousands of charts.

## Troubleshooting

### "Window doesn't appear"
- Press Cmd+Tab (Mac) or Alt+Tab (Windows) to find the Polyscope window
- Try Mission Control (3-finger swipe up on Mac) to see all windows
- Check if window is on another desktop/space

### "Script hangs during atlas generation"
- This is normal for large meshes (Bunny: ~2min, Armadillo: ~5min)
- Check terminal for progress messages
- For quick testing, use Spot mesh (smaller, ~30 seconds)

### "Too many charts to display"
- Edit `MAX_CHARTS_TO_SHOW` in `visualize_atlas.py` to reduce number of displayed charts
- Large meshes (Bunny: 4,583 charts) may be slow to navigate

## What to Look For

### Good Segmentation:
- Charts follow natural features (creases, high curvature areas)
- Chart boundaries align with sharp edges
- Charts are roughly similar sizes
- Most faces belong to medium/large charts (not tiny fragments)

### Good LSCM Parameterization:
- Low angle distortion (close to 1.0 = perfect conformal)
- Minimal fold-overs (0 is ideal)
- UV layout is smooth without severe stretching
- Checkerboard texture looks regular (not highly distorted)

### Our Results Show:
- ✅ 96-98% success rate on complex meshes
- ✅ Angle distortion 1.000-1.022 (excellent)
- ✅ Most charts have 0 fold-overs
- ✅ Automatic segmentation creates meaningful chart boundaries

## Technical Details

### How It Works:

1. **Feature Detection** (visualize_charts.py shows this)
   - Compute edge sharpness (angle between face normals)
   - Keep top 5% sharpest edges as features
   - Grow feature curves along sharp edges

2. **Chart Growing** (colored regions show this)
   - Compute distance from each face to nearest feature
   - Place seeds at local maxima of distance-to-features
   - Grow charts using modified Dijkstra algorithm
   - Merge nearby charts

3. **LSCM Parameterization** (visualize_atlas.py shows this)
   - For each connected chart component
   - Solve sparse least-squares system for UV coordinates
   - Minimize Cauchy-Riemann violation (conformal energy)
   - Validate results (check topology, distortion, overlaps)

### Performance:
- Feature detection: O(E log E) where E = edges
- Chart growing: O(F log F) where F = faces
- LSCM per chart: O(n) for n vertices (sparse solver)
- Total: Approximately linear in mesh size

## For Your Demo/Presentation

**Recommended workflow:**
1. Run `visualize_charts.py` on Spot → Show automatic segmentation
2. Run `visualize_atlas.py` on Spot → Show individual charts with LSCM
3. Use slider to show different charts and their parameterizations
4. Toggle "Show 3D" checkbox to demonstrate UV flattening

**Key points to highlight:**
- Automatic segmentation finds natural chart boundaries
- LSCM produces near-perfect conformal maps (distortion ~1.0)
- System handles complex topology (Bunny has χ=-3)
- 96-98% success rate demonstrates robustness

## Files Generated

Running the visualization scripts does NOT modify any files. To save results:
- Use screenshot feature in Polyscope (camera icon or File menu)
- Run test scripts to generate text reports: `python test_atlas.py > results.txt`
