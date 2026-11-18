"""Shared PathSnap data pipeline. No evaluator data is discovered automatically.

Candidate features never read .route. Reference alignment is a separate,
offline operation producing weak observation labels, not measured ground truth.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pyproj import CRS, Transformer

# This file supports 01_data_exploration.ipynb:
# 1. Read and check each trip.
# 2. Find road segments near each GPS point and describe each possible match.
# 3. Separately compare GPS points with the supplied route to make rough labels.
# 4. Save the processed data and draw an example map.
# The model-training and route-reconstruction notebooks are still outlines.

# The order here is the column order in the saved candidate feature matrix.
# One row describes one possible match between a GPS point and a road arc.
FEATURE_NAMES = [
    "distance_m", "arc_length_m", "projection_fraction",
    "distance_to_start_m", "distance_to_end_m", "heading_cos",
    "heading_sin", "heading_valid", "speed_mps", "dt_s",
    "step_distance_m", "gap_flag", "speed_outlier_flag",
    "previous_point_distance_m", "candidate_rank", "candidate_count",
    "distance_margin_m", "out_degree", "in_degree",
]

@dataclass(frozen=True)
class Config:
    """Settings shared by the steps in notebook 01.

    Search up to 100 m around a GPS point and keep at most eight road arcs.
    The remaining values flag unusual GPS motion or uncertain rough labels.
    """
    candidate_radius_m: float = 100.0
    max_candidates: int = 8
    index_spacing_m: float = 50.0
    gap_seconds: float = 30.0
    max_plausible_speed_mps: float = 60.0
    heading_min_step_m: float = 3.0
    label_max_distance_m: float = 50.0
    label_max_monotonic_excess_m: float = 25.0

def discover_trips(data_root, partition):
    """List trip folders in train or student_test.

    Used by notebook 01 when counting trips, choosing an example and making
    validation folds. The export step also calls it for every trip.
    """
    if partition not in ("train", "student_test"):
        raise ValueError("Only train and student_test are permitted in development.")
    folder = Path(data_root) / partition
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(p for p in folder.iterdir() if p.is_dir())

def read_numeric(path, columns):
    """Read a whitespace-separated file and check its column count.

    The .track, .nodes, .arcs and .route files have no column headers.
    Other file-reading functions call this helper.
    """
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, got {values.shape}")
    return values

def read_graph(trip):
    """Read the road network for one trip.

    .nodes contains road-point coordinates. Each .arcs row connects two node
    IDs; that row's position is its arc ID. Notebook 01 uses the network to
    find candidates, measure features and draw grey road lines.
    """
    trip = Path(trip)
    nodes = read_numeric(trip / f"{trip.name}.nodes", 2)
    raw = read_numeric(trip / f"{trip.name}.arcs", 2)
    if not np.isfinite(nodes).all() or np.any(np.abs(nodes) > [180, 90]):
        raise ValueError(f"{trip}: invalid node coordinates")
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError(f"{trip}: arc IDs must be finite integers")
    arcs = raw.astype(np.int64)
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f"{trip}: arc endpoint outside node table")
    # Never reorder arcs: their row numbers are the IDs used by .route.
    return nodes, arcs

def read_reference(trip, n_arcs):
    """Read the supplied sequence of arc IDs from .route.

    Notebook 01 uses it to draw the green path and make rough training labels.
    It is deliberately not read while building candidate features.
    """
    trip = Path(trip)
    raw = read_numeric(trip / f"{trip.name}.route", 1).ravel()
    if not len(raw) or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError(f"{trip}: invalid reference route")
    route = raw.astype(np.int64)
    if route.min() < 0 or route.max() >= n_arcs:
        raise ValueError(f"{trip}: reference arc outside graph")
    return route

def clean_track(trip, config=Config()):
    """Read .track and return GPS points in time order plus a cleaning report.

    Bad coordinates/timestamps, gaps over 30 s and apparent speeds over
    60 m/s are removed; duplicate times keep their first row.
    observation_id preserves the row's original position in .track.
    Notebook 01 uses this for checks, features, exports and maps.
    """
    trip = Path(trip)
    raw = read_numeric(trip / f"{trip.name}.track", 3)
    frame = pd.DataFrame(raw, columns=["longitude", "latitude", "timestamp_s"])
    frame.insert(0, "observation_id", np.arange(len(frame)))
    valid = np.isfinite(raw).all(axis=1) & (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    clean = frame.loc[valid].sort_values("timestamp_s", kind="stable")
    duplicate = clean.duplicated("timestamp_s", keep="first")
    audit = {
        "raw_observations": len(raw),
        "invalid_observations": int((~valid).sum()),
        "duplicate_timestamps_removed": int(duplicate.sum()),
        "original_nonincreasing_steps": int(np.sum(np.diff(raw[:, 2]) <= 0)),
    }
    clean = clean.loc[~duplicate].reset_index(drop=True)
    if clean.empty:
        raise ValueError(f"{trip}: no valid observations")
    # Judge each row against its previous row in the original time-sorted track.
    # Remove flagged rows once; do not repeatedly delete points after a GPS gap.
    lon, lat = clean[["longitude", "latitude"]].median()
    _, transformer = projection_for(np.array([[lon, lat]]))
    xy = project(transformer, clean[["longitude", "latitude"]].to_numpy())
    dt = np.r_[0.0, np.diff(clean.timestamp_s.to_numpy())]
    step = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
    gap_flag = dt > config.gap_seconds
    speed_flag = speed > config.max_plausible_speed_mps
    audit["gap_observations_removed"] = int(gap_flag.sum())
    audit["speed_observations_removed"] = int(speed_flag.sum())
    audit["motion_observations_removed"] = int((gap_flag | speed_flag).sum())
    clean = clean.loc[~(gap_flag | speed_flag)].reset_index(drop=True)
    if clean.empty:
        raise ValueError(f"{trip}: no observations after motion filtering")
    clean.insert(0, "trip_id", trip.name)
    audit["clean_observations"] = len(clean)
    return clean, audit

def projection_for(nodes):
    """Choose a local map projection from the trip's road coordinates.

    GPS longitude/latitude are angles. The projection lets us measure road
    lengths and GPS-to-road distances in metres instead of degrees.
    """
    lon, lat = np.median(nodes, axis=0)
    zone = int(np.clip(np.floor((lon + 180) / 6) + 1, 1, 60))
    epsg = (32600 if lat >= 0 else 32700) + zone
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    return epsg, transformer

def project(transformer, coordinates):
    """Convert longitude/latitude pairs into x/y coordinates in metres."""
    x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
    return np.column_stack([x, y])

def segment_geometry(point, starts, vectors, length_sq):
    """Measure a point's distance to each road segment.

    Also return where the nearest position falls along the segment:
    0 is its start node and 1 is its end node.
    """
    fraction = np.clip(np.sum((point - starts) * vectors, axis=1) / np.maximum(length_sq, 1e-12), 0, 1)
    nearest = starts + fraction[:, None] * vectors
    distance = np.linalg.norm(point - nearest, axis=1)
    return distance, fraction, nearest

def segment_samples(starts, vectors, lengths, arc_ids, spacing):
    """Place search points along roads so long arcs can still be found."""
    intervals = np.maximum(1, np.ceil(lengths[arc_ids] / spacing).astype(int))
    counts = intervals + 1
    sample_arcs = np.repeat(arc_ids, counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    fractions = (np.arange(counts.sum()) - offsets) / np.repeat(intervals, counts)
    points = starts[sample_arcs] + fractions[:, None] * vectors[sample_arcs]
    return sample_arcs, points

class RoadIndex:
    """Find road segments close to a GPS point.

    Sample points along each road segment to search quickly. Then calculate
    exact point-to-segment distances before deciding which arcs are candidates.
    Notebook 01 creates this index for its example and for every exported trip.
    """
    def __init__(self, nodes, arcs, config=Config()):
        """Project nodes, index nonzero road arcs and count graph connections."""
        self.config = config
        self.nodes, self.arcs = nodes, arcs
        self.epsg, self.transformer = projection_for(nodes)
        self.xy = project(self.transformer, nodes)
        self.starts = self.xy[arcs[:, 0]]
        self.vectors = self.xy[arcs[:, 1]] - self.starts
        self.length_sq = np.sum(self.vectors ** 2, axis=1)
        self.length = np.sqrt(self.length_sq)
        self.valid_arcs = np.flatnonzero(self.length > 1e-6)
        if not len(self.valid_arcs):
            raise ValueError("Graph contains no nonzero road segments")
        self.sample_arc, samples = segment_samples(
            self.starts, self.vectors, self.length,
            self.valid_arcs, config.index_spacing_m
        )
        self.tree = cKDTree(samples)
        self.out_degree = np.bincount(arcs[:, 0], minlength=len(nodes))
        self.in_degree = np.bincount(arcs[:, 1], minlength=len(nodes))

    def candidates(self, point):
        """Return the closest arcs within the configured search radius.

        build_candidate_features calls this once for each cleaned GPS point.
        An empty result stays empty so candidate misses can be counted.
        """
        c = self.config
        samples = self.tree.query_ball_point(point, c.candidate_radius_m + c.index_spacing_m / 2 + 1e-6)
        ids = np.unique(self.sample_arc[samples])
        if len(ids):
            distance, fraction, nearest = segment_geometry(point, self.starts[ids], self.vectors[ids], self.length_sq[ids])
            keep = distance <= c.candidate_radius_m + 1e-6
            ids, distance, fraction, nearest = ids[keep], distance[keep], fraction[keep], nearest[keep]
            order = np.lexsort((ids, distance))[:c.max_candidates]
            return ids[order], distance[order], fraction[order], nearest[order]
        return np.array([], dtype=np.int64), np.array([]), np.array([]), np.empty((0, 2))

def motion_features(clean, road, config):
    """Describe how the vehicle moved since the previous GPS point.

    Produces heading, elapsed time, distance, speed and gap/outlier flags.
    The feature builder calls this; it does not look at future GPS points.
    """
    xy = project(road.transformer, clean[["longitude", "latitude"]].to_numpy())
    delta = np.vstack([np.zeros((1, 2)), np.diff(xy, axis=0)])
    step = np.linalg.norm(delta, axis=1)
    dt = np.r_[0.0, np.diff(clean.timestamp_s.to_numpy())]
    speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
    gap = dt > config.gap_seconds
    outlier = speed > config.max_plausible_speed_mps
    heading_valid = (step >= config.heading_min_step_m) & (~gap) & (~outlier) & (dt > 0)
    direction = np.divide(delta, step[:, None], out=np.zeros_like(delta), where=step[:, None] > 0)
    return xy, direction, step, dt, speed, gap, outlier, heading_valid

def features_for_one_point(i, point, ids, distances, fractions, xy, direction,
                           step, dt, speed, gap, outlier, heading_valid, road):
    """Describe one GPS point against all its nearby road arcs."""
    length = road.length[ids]
    unit = road.vectors[ids] / length[:, None]
    n = len(ids)
    if heading_valid[i]:
        heading_cos = unit @ direction[i]
        heading_sin = direction[i, 0] * unit[:, 1] - direction[i, 1] * unit[:, 0]
    else:
        heading_cos = heading_sin = np.zeros(n)
    previous_point = xy[max(0, i - 1)]
    previous_distance = segment_geometry(
        previous_point, road.starts[ids], road.vectors[ids], road.length_sq[ids]
    )[0]
    end_points = road.starts[ids] + road.vectors[ids]
    return np.column_stack([
        distances, length, fractions,
        np.linalg.norm(point - road.starts[ids], axis=1),
        np.linalg.norm(point - end_points, axis=1),
        heading_cos, heading_sin, np.full(n, heading_valid[i]),
        np.full(n, speed[i]), np.full(n, dt[i]), np.full(n, step[i]),
        np.full(n, gap[i]), np.full(n, outlier[i]), previous_distance,
        np.arange(n), np.full(n, n), distances - distances[0],
        road.out_degree[road.arcs[ids, 0]],
        road.in_degree[road.arcs[ids, 1]],
    ]).astype(np.float32)

def build_candidate_features(clean, road, config=Config()):
    """Make one 19-feature row per GPS point and nearby road candidate.

    Distance and geometry describe how the point fits the road. Motion
    describes whether the road direction fits recent travel. Return the
    feature matrix, the matching GPS/arc IDs and observation diagnostics.
    Notebook 01 calls this for an example and for all-trip export.
    This function never reads .route or creates labels.
    """
    xy, direction, step, dt, speed, gap, outlier, heading_valid = motion_features(clean, road, config)
    blocks, observation_indices, arc_ids = [], [], []
    counts = np.zeros(len(clean), dtype=np.int32)
    for i, point in enumerate(xy):
        ids, dist, fraction, _ = road.candidates(point)
        counts[i] = len(ids)
        if not len(ids):
            continue  # Record a miss; do not pretend a distant road is nearby.
        blocks.append(features_for_one_point(
            i, point, ids, dist, fraction, xy, direction, step, dt, speed,
            gap, outlier, heading_valid, road
        ))
        observation_indices.extend([i] * len(ids))
        arc_ids.extend(ids)
    X = np.concatenate(blocks) if blocks else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite candidate features")
    observations = clean.copy()
    observations["x_m"], observations["y_m"] = xy.T
    observations["dt_s"], observations["step_distance_m"] = dt, step
    observations["speed_mps"], observations["gap_flag"] = speed, gap
    observations["speed_outlier_flag"] = outlier
    observations["candidate_count"] = counts
    return dict(X=X, observation_index=np.asarray(observation_indices, dtype=np.int32),
                arc_id=np.asarray(arc_ids, dtype=np.int64)), observations

def choose_route_positions(points, starts, vectors, length_sq):
    """Match GPS points to an ordered route without moving backward.

    For each point, keep the cheapest earlier route position. Work backward
    at the end to recover the chosen position for every point.
    """
    count, route_count = len(points), len(starts)
    previous = np.zeros(route_count)
    positions = np.arange(route_count)
    back = np.empty((count, route_count), dtype=np.uint32)
    nearest = np.empty(count)

    for i, point in enumerate(points):
        distances = segment_geometry(point, starts, vectors, length_sq)[0]
        nearest[i] = distances.min()
        best_so_far = np.minimum.accumulate(previous)
        best_position = np.maximum.accumulate(
            np.where(previous <= best_so_far, positions, 0)
        )
        back[i] = best_position
        previous = best_so_far + distances ** 2
        previous -= previous.min()  # Keep numbers small; the best path is unchanged.

    chosen = np.empty(count, dtype=np.int32)
    chosen[-1] = int(np.argmin(previous))
    for i in range(count - 1, 0, -1):
        chosen[i - 1] = back[i, chosen[i]]
    return chosen, nearest

def align_reference(clean, road, route, config=Config()):
    """Estimate a reference arc for each GPS point for later model training.

    .route lists travelled arcs, but has no GPS-row labels. Follow its order
    while choosing the closest plausible route position for each GPS point.
    Points may remain on one arc or move forward; they cannot move backward.
    Flag estimates that are too far from the road or otherwise uncertain.
    Notebook 01 uses these *rough labels* for analysis and saved targets.
    This is not the future HMM prediction decoder.
    """
    xy = project(road.transformer, clean[["longitude", "latitude"]].to_numpy())
    starts, vectors, sq = road.starts[route], road.vectors[route], road.length_sq[route]
    selected, nearest_distance = choose_route_positions(xy, starts, vectors, sq)
    chosen = route[selected]
    distance = np.empty(len(xy))
    for i in range(len(xy)):
        distance[i] = segment_geometry(xy[i], road.starts[chosen[i:i+1]], road.vectors[chosen[i:i+1]], road.length_sq[chosen[i:i+1]])[0][0]
    connected = bool(np.all(road.arcs[route[:-1], 1] == road.arcs[route[1:], 0]))
    reliable = ((distance <= config.label_max_distance_m)
                & (distance - nearest_distance <= config.label_max_monotonic_excess_m)
                & (road.length[chosen] > 1e-6) & connected)
    return pd.DataFrame({
        "observation_id": clean.observation_id.to_numpy(),
        "reference_arc_id": chosen, "route_position": selected,
        "alignment_distance_m": distance, "nearest_reference_distance_m": nearest_distance,
        "label_usable": reliable, "reference_connected": connected,
        "label_source": "monotone_reference_alignment_weak",
    })

def inventory(data_root):
    """Summarise every permitted trip for notebook 01's first table.

    Count GPS rows, roads, route entries, time gaps and graph problems.
    """
    rows = []
    for partition in ("train", "student_test"):
        for trip in discover_trips(data_root, partition):
            required = [trip / f"{trip.name}.{ext}" for ext in ("track", "nodes", "arcs", "route")]
            row = {"partition": partition, "trip_id": trip.name,
                   "missing_files": ",".join(p.suffix for p in required if not p.exists()),
                   "file_bytes": sum(p.stat().st_size for p in required if p.exists())}
            try:
                clean, audit = clean_track(trip)
                nodes, arcs = read_graph(trip)
                route = read_reference(trip, len(arcs))
                epsg, transformer = projection_for(nodes)
                xy = project(transformer, nodes)
                lengths = np.linalg.norm(xy[arcs[:, 1]] - xy[arcs[:, 0]], axis=1)
                adjacency = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
                n_components, _ = connected_components(adjacency, directed=True, connection="weak")
                degree = np.bincount(arcs.ravel(), minlength=len(nodes))
                dt = np.diff(clean.timestamp_s.to_numpy())
                gps_xy = project(transformer, clean[["longitude", "latitude"]].to_numpy())
                steps = np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)
                row.update(audit)
                row.update(nodes=len(nodes), arcs=len(arcs), route_entries=len(route), epsg=epsg,
                           weak_components=n_components, isolated_nodes=int((degree == 0).sum()),
                           zero_length_arcs=int((lengths <= 1e-6).sum()),
                           duplicate_arcs=int(len(arcs) - len(np.unique(arcs, axis=0))),
                           route_breaks=int(np.sum(arcs[route[:-1], 1] != arcs[route[1:], 0])),
                           duration_s=float(clean.timestamp_s.iloc[-1] - clean.timestamp_s.iloc[0]),
                           median_dt_s=float(np.median(dt)) if len(dt) else 0,
                           max_dt_s=float(dt.max()) if len(dt) else 0,
                           one_second_steps=int(np.sum(np.isclose(dt, 1))),
                           time_steps=len(dt), gap_count=int(np.sum(dt > Config().gap_seconds)),
                           gps_length_m=float(steps.sum()),
                           speed_outliers=int(np.sum(steps / dt > Config().max_plausible_speed_mps)),
                           min_lon=float(clean.longitude.min()), max_lon=float(clean.longitude.max()),
                           min_lat=float(clean.latitude.min()), max_lat=float(clean.latitude.max()),
                           status="ok")
            except (ValueError, OSError) as exc:
                row.update(status="error", error=str(exc))
            rows.append(row)
    return pd.DataFrame(rows)

def source_hashes(trip):
    """Fingerprint raw files so an export can be tied to its exact inputs."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(trip).iterdir()) if p.suffix in (".track", ".nodes", ".arcs", ".route")}

def process_dataset(data_root, output_root, config=Config()):
    """Run the data pipeline for all train and student_test trips.

    Save candidate features, cleaned GPS points, separate rough labels and a
    summary of coverage. Notebook 01 calls this in its export section.
    """
    output_root = Path(output_root)
    summary = []
    for partition in ("train", "student_test"):
        trips = discover_trips(data_root, partition)
        for number, trip in enumerate(trips, 1):
            started = time.perf_counter()
            clean, audit = clean_track(trip, config)
            nodes, arcs = read_graph(trip)
            road = RoadIndex(nodes, arcs, config)
            candidates, observations = build_candidate_features(clean, road, config)
            # Make features first; only then read .route to create labels.
            route = read_reference(trip, len(arcs))
            labels = align_reference(clean, road, route, config)
            oi = candidates["observation_index"]
            y = (candidates["arc_id"] == labels.reference_arc_id.to_numpy()[oi]).astype(np.int8)
            usable = labels.label_usable.to_numpy()
            hits = np.zeros(len(clean), dtype=bool)
            hits[oi[y == 1]] = True
            # If the reference arc is missing from a point's candidate list,
            # do not train on that all-negative group. Still count the miss.
            fit_mask = usable[oi] & hits[oi]
            target = output_root / partition / trip.name
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / "candidates.npz", **candidates,
                                feature_names=np.asarray(FEATURE_NAMES))
            np.savez_compressed(target / "targets.npz", y=y, fit_mask=fit_mask)
            observations.to_csv(target / "observations.csv.gz", index=False)
            labels.assign(reference_in_candidates=hits).to_csv(target / "labels.csv.gz", index=False)
            metadata = {"partition": partition, "trip_id": trip.name, "epsg": road.epsg,
                        "config": asdict(config), "source_sha256": source_hashes(trip),
                        "label_type": "derived_weak_labels", "feature_names": FEATURE_NAMES}
            (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            row = dict(partition=partition, trip_id=trip.name, **audit,
                       candidate_rows=len(y), no_candidate_observations=int((observations.candidate_count == 0).sum()),
                       usable_label_observations=int(usable.sum()),
                       reference_candidate_misses=int((usable & ~hits).sum()),
                       candidate_miss_rate=float((usable & ~hits).sum() / max(1, usable.sum())),
                       fit_rows=int(fit_mask.sum()), positive_fit_rows=int((y[fit_mask] == 1).sum()),
                       alignment_median_m=float(labels.alignment_distance_m.median()),
                       alignment_p95_m=float(labels.alignment_distance_m.quantile(.95)),
                       seconds=round(time.perf_counter() - started, 3))
            summary.append(row)
            if number % 10 == 0 or number == len(trips):
                print(f"{partition}: {number}/{len(trips)} trips processed", flush=True)
    result = pd.DataFrame(summary)
    result.to_csv(output_root / "processing_summary.csv", index=False)
    manifest = {"config": asdict(config), "feature_names": FEATURE_NAMES,
                "python": platform.python_version(),
                "partitions": {p: [t.name for t in discover_trips(data_root, p)] for p in ("train", "student_test")},
                "label_caveat": "Pointwise targets are derived alignments, not independently supplied GPS labels.",
                "scaling": "No learned scaling fitted. Fit transformers inside training CV folds only."}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result

def make_map(trip, output_path, max_gps_points=1500):
    """Draw notebook 01's interactive example map.

    Blue circles are GPS points from .track, grey lines are network arcs,
    and green lines are the supplied .route. No model prediction is shown.
    The map uses no external background tiles, so blocked tile requests
    cannot cover the trip data.
    """
    import folium
    trip = Path(trip)
    clean, _ = clean_track(trip)
    nodes, arcs = read_graph(trip)
    route = read_reference(trip, len(arcs))
    center = [clean.latitude.median(), clean.longitude.median()]
    map_ = folium.Map(location=center, zoom_start=13, tiles=None)
    low = clean[["longitude", "latitude"]].min().to_numpy() - .003
    high = clean[["longitude", "latitude"]].max().to_numpy() + .003
    a, b = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    keep = np.all(np.maximum(a, b) >= low, axis=1) & np.all(np.minimum(a, b) <= high, axis=1)
    road_group = folium.FeatureGroup(name="Road network in GPS bounds", show=False)
    geometry = [[[float(a[i, 0]), float(a[i, 1])], [float(b[i, 0]), float(b[i, 1])]] for i in np.flatnonzero(keep)]
    folium.GeoJson({"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": geometry}},
                   style_function=lambda _: {"color": "#8b95a5", "weight": 1, "opacity": .5}).add_to(road_group)
    road_group.add_to(map_)
    reference_group = folium.FeatureGroup(name="Supplied reference route (development only)")
    for arc_id in route:
        folium.PolyLine(nodes[arcs[arc_id]][:, ::-1].tolist(), color="#16a34a", weight=4, opacity=.8).add_to(reference_group)
    reference_group.add_to(map_)
    gps_group = folium.FeatureGroup(name="GPS observations (blue)")
    stride = max(1, int(np.ceil(len(clean) / max_gps_points)))
    for row in clean.iloc[::stride].itertuples():
        folium.CircleMarker([row.latitude, row.longitude], radius=2, color="#2563eb",
                            fill=True, tooltip=f"GPS row {row.observation_id}, t={row.timestamp_s}s").add_to(gps_group)
    gps_group.add_to(map_)
    folium.LayerControl().add_to(map_)
    map_.fit_bounds([[float(clean.latitude.min()), float(clean.longitude.min())],
                     [float(clean.latitude.max()), float(clean.longitude.max())]])
    map_.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:25px;left:25px;background:white;padding:12px;z-index:9999">'
        '<b>PathSnap development EDA</b><br>Blue: GPS observations<br>Green: supplied reference'
        '<br>Grey: road network (toggle layer)<br>No predicted route: models not trained yet.</div>'))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_.save(str(output_path))
    return map_

