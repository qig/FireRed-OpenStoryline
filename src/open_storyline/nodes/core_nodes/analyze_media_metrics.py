from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_schema import (
    AnalyzeAudioEnergyInput,
    AnalyzeBeatRhythmInput,
    AnalyzeMotionEnergyInput,
)
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.register import NODE_REGISTRY


def _find_project_root() -> Path | None:
    starts: list[Path] = []
    env_root = os.getenv("KOKOMO_PROJECT_ROOT", "").strip()
    if env_root:
        starts.append(Path(env_root).expanduser().resolve())
    starts.append(Path.cwd().resolve())
    starts.append(Path(__file__).resolve())

    for start in starts:
        cursor = start if start.is_dir() else start.parent
        for candidate in [cursor, *cursor.parents]:
            if (candidate / "MotionScoring").is_dir() and (candidate / "BeatTracking").is_dir():
                return candidate
    return None


def _ensure_project_root_on_path() -> None:
    root = _find_project_root()
    if root is None:
        return
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _abs_path(raw_path: Any) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _sample_indices(total: int, max_points: int) -> list[int]:
    if total <= 0:
        return []
    if max_points <= 0 or total <= max_points:
        return list(range(total))
    if max_points == 1:
        return [0]
    lin = np.linspace(0, total - 1, max_points)
    return [int(round(x)) for x in lin.tolist()]


def _sample_series(values: Sequence[Any], max_points: int) -> list[Any]:
    arr = list(values)
    if len(arr) <= max_points:
        return arr
    idx = _sample_indices(len(arr), max_points)
    return [arr[i] for i in idx]


def _collect_video_clips(
    inputs: Dict[str, Any],
    max_clips: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    clips = (inputs.get("split_shots") or {}).get("clips") or []
    selected: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "non_video": 0,
        "missing_path": 0,
        "path_not_found": 0,
    }
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        stats["total"] += 1

        if str(clip.get("kind", "")).lower() != "video":
            stats["non_video"] += 1
            continue

        path_text = str(clip.get("path", "")).strip()
        if not path_text:
            stats["missing_path"] += 1
            continue

        clip_path = _abs_path(path_text)
        if not clip_path.exists():
            stats["path_not_found"] += 1
            continue

        selected.append(
            {
                "clip_id": str(clip.get("clip_id", "")),
                "path": str(clip_path),
                "source_ref": clip.get("source_ref") or {},
            }
        )
        if max_clips > 0 and len(selected) >= max_clips:
            break

    return selected, stats


@NODE_REGISTRY.register()
class AnalyzeMotionEnergyNode(BaseNode):
    meta = NodeMeta(
        name="analyze_motion_energy",
        description=(
            "Analyze per-clip visual motion energy using dense optical flow. "
            "Useful for dynamic pacing, hook strength, and energy-curve matching."
        ),
        node_id="analyze_motion_energy",
        node_kind="analyze_motion_energy",
        require_prior_kind=["split_shots"],
        default_require_prior_kind=["split_shots"],
        next_available_node=["filter_clips", "group_clips", "plan_timeline_pro"],
    )
    input_schema = AnalyzeMotionEnergyInput

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        node_state.node_summary.info_for_user("Skipped motion-energy analysis")
        return {
            "motion_energy": {
                "clips": [],
                "summary": {"analyzed_video_clips": 0, "status": "skipped"},
            }
        }

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        _ensure_project_root_on_path()
        try:
            from MotionScoring.motion_scorer import MotionScorer
        except Exception as e:
            node_state.node_summary.add_error(f"MotionScoring import failed: {e}")
            return {
                "motion_energy": {
                    "clips": [],
                    "summary": {"analyzed_video_clips": 0, "import_error": str(e)},
                }
            }

        max_clips = int(inputs.get("max_clips", 0) or 0)
        selected, stats = _collect_video_clips(inputs, max_clips=max_clips)
        if not selected:
            node_state.node_summary.info_for_user("No valid video clips for motion-energy analysis")
            return {
                "motion_energy": {
                    "clips": [],
                    "summary": {
                        "analyzed_video_clips": 0,
                        "total_clips": stats["total"],
                        "skipped_non_video": stats["non_video"],
                        "skipped_missing_path": stats["missing_path"] + stats["path_not_found"],
                    },
                }
            }

        scorer = MotionScorer(
            flow_method=str(inputs.get("flow_method", "farneback")),
            resize_width=int(inputs.get("resize_width", 480)),
            frame_skip=int(inputs.get("frame_skip", 1)),
        )

        results: list[dict[str, Any]] = []
        failed = 0
        for item in selected:
            clip_id = item.get("clip_id", "")
            clip_path = item.get("path", "")
            try:
                analysis = scorer.analyze(clip_path)
                motion_score = float(analysis.global_score)
                results.append(
                    {
                        "clip_id": clip_id,
                        "path": clip_path,
                        "duration_ms": int((item.get("source_ref") or {}).get("duration", 0) or 0),
                        "motion_score": round(motion_score, 6),
                        "flow_method": analysis.flow_method,
                        "analysis_fps": round(float(analysis.fps), 3),
                        "frame_count": int(analysis.total_frames),
                    }
                )
            except Exception as e:
                failed += 1
                node_state.node_summary.add_warning(f"[motion] {clip_id} failed: {e}")

        if results:
            scores = [float(row["motion_score"]) for row in results]
            s_min, s_max = min(scores), max(scores)
            score_span = s_max - s_min
            for row in results:
                score = float(row["motion_score"])
                if score_span <= 1e-9:
                    row["motion_score_norm"] = 0.5
                else:
                    row["motion_score_norm"] = round((score - s_min) / score_span, 4)

        node_state.node_summary.info_for_user(
            f"Motion-energy analysis completed: {len(results)} clip(s) analyzed, {failed} failed."
        )
        return {
            "motion_energy": {
                "clips": results,
                "summary": {
                    "analyzed_video_clips": len(results),
                    "failed_video_clips": failed,
                    "total_clips": stats["total"],
                    "skipped_non_video": stats["non_video"],
                    "skipped_missing_path": stats["missing_path"] + stats["path_not_found"],
                },
            }
        }


@NODE_REGISTRY.register()
class AnalyzeBeatRhythmNode(BaseNode):
    meta = NodeMeta(
        name="analyze_beat_rhythm",
        description=(
            "Analyze beat/rhythm structure from each clip's audio. "
            "Outputs BPM, beat timestamps, and trigger points for rhythm-driven edits."
        ),
        node_id="analyze_beat_rhythm",
        node_kind="analyze_beat_rhythm",
        require_prior_kind=["split_shots"],
        default_require_prior_kind=["split_shots"],
        next_available_node=["group_clips", "plan_timeline_pro", "render_video"],
    )
    input_schema = AnalyzeBeatRhythmInput

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        node_state.node_summary.info_for_user("Skipped beat/rhythm analysis")
        return {
            "beat_rhythm": {
                "clips": [],
                "summary": {"analyzed_video_clips": 0, "status": "skipped"},
            }
        }

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        _ensure_project_root_on_path()
        try:
            from BeatTracking.beat_analyzer import BeatAnalyzer
            from FootageAnalysis.FootageAnalyzer.audio_energy import _extract_audio
        except Exception as e:
            node_state.node_summary.add_error(f"BeatTracking import failed: {e}")
            return {
                "beat_rhythm": {
                    "clips": [],
                    "summary": {"analyzed_video_clips": 0, "import_error": str(e)},
                }
            }

        sample_rate = int(inputs.get("sample_rate", 22050))
        start_bpm = float(inputs.get("start_bpm", 120.0))
        tightness = float(inputs.get("tightness", 100.0))
        beats_per_bar = int(inputs.get("beats_per_bar", 4))
        max_points = int(inputs.get("max_points", 120))
        max_clips = int(inputs.get("max_clips", 0) or 0)

        selected, stats = _collect_video_clips(inputs, max_clips=max_clips)
        if not selected:
            node_state.node_summary.info_for_user("No valid video clips for beat/rhythm analysis")
            return {
                "beat_rhythm": {
                    "clips": [],
                    "summary": {
                        "analyzed_video_clips": 0,
                        "total_clips": stats["total"],
                        "skipped_non_video": stats["non_video"],
                        "skipped_missing_path": stats["missing_path"] + stats["path_not_found"],
                    },
                }
            }

        analyzer = BeatAnalyzer(
            sr=sample_rate,
            start_bpm=start_bpm,
            tightness=tightness,
            beats_per_bar=beats_per_bar,
        )

        results: list[dict[str, Any]] = []
        failed = 0
        for item in selected:
            clip_id = item.get("clip_id", "")
            clip_path = item.get("path", "")
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    wav_path = _extract_audio(clip_path, tmp_dir, sr=sample_rate)
                    analysis = analyzer.analyze(wav_path)

                beat_times_ms = [int(round(float(t) * 1000.0)) for t in analysis.beat_times.tolist()]
                downbeat_times_ms = [int(round(float(t) * 1000.0)) for t in analysis.downbeat_times.tolist()]
                hard_cut_ms = [int(round(float(t) * 1000.0)) for t in analysis.get_hard_cut_triggers().tolist()]
                flash_ms = [int(round(float(t) * 1000.0)) for t in analysis.get_flash_triggers().tolist()]

                results.append(
                    {
                        "clip_id": clip_id,
                        "path": clip_path,
                        "tempo_bpm": round(float(analysis.tempo), 3),
                        "beats_per_bar": int(analysis.beats_per_bar),
                        "beat_count": len(beat_times_ms),
                        "beat_times_ms": _sample_series(beat_times_ms, max_points),
                        "downbeat_times_ms": _sample_series(downbeat_times_ms, max_points),
                        "hard_cut_triggers_ms": _sample_series(hard_cut_ms, max_points),
                        "flash_triggers_ms": _sample_series(flash_ms, max_points),
                    }
                )
            except Exception as e:
                failed += 1
                node_state.node_summary.add_warning(f"[beat] {clip_id} failed: {e}")

        node_state.node_summary.info_for_user(
            f"Beat/rhythm analysis completed: {len(results)} clip(s) analyzed, {failed} failed."
        )
        return {
            "beat_rhythm": {
                "clips": results,
                "summary": {
                    "analyzed_video_clips": len(results),
                    "failed_video_clips": failed,
                    "total_clips": stats["total"],
                    "skipped_non_video": stats["non_video"],
                    "skipped_missing_path": stats["missing_path"] + stats["path_not_found"],
                },
            }
        }


@NODE_REGISTRY.register()
class AnalyzeAudioEnergyNode(BaseNode):
    meta = NodeMeta(
        name="analyze_audio_energy",
        description=(
            "Analyze per-clip audio energy curves (RMS, low-band, high-band). "
            "Useful for pacing, beat emphasis, and audio-visual energy alignment."
        ),
        node_id="analyze_audio_energy",
        node_kind="analyze_audio_energy",
        require_prior_kind=["split_shots"],
        default_require_prior_kind=["split_shots"],
        next_available_node=["group_clips", "plan_timeline_pro", "render_video"],
    )
    input_schema = AnalyzeAudioEnergyInput

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        node_state.node_summary.info_for_user("Skipped audio-energy analysis")
        return {
            "audio_energy": {
                "clips": [],
                "summary": {"analyzed_video_clips": 0, "status": "skipped"},
            }
        }

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        _ensure_project_root_on_path()
        try:
            from FootageAnalysis.FootageAnalyzer.audio_energy import _extract_audio, compute_audio_energy
        except Exception as e:
            node_state.node_summary.add_error(f"AudioEnergy import failed: {e}")
            return {
                "audio_energy": {
                    "clips": [],
                    "summary": {"analyzed_video_clips": 0, "import_error": str(e)},
                }
            }

        sample_rate = int(inputs.get("sample_rate", 22050))
        n_fft = int(inputs.get("n_fft", 2048))
        hop_length = int(inputs.get("hop_length", 512))
        max_points = int(inputs.get("max_points", 120))
        max_clips = int(inputs.get("max_clips", 0) or 0)

        selected, stats = _collect_video_clips(inputs, max_clips=max_clips)
        if not selected:
            node_state.node_summary.info_for_user("No valid video clips for audio-energy analysis")
            return {
                "audio_energy": {
                    "clips": [],
                    "summary": {
                        "analyzed_video_clips": 0,
                        "total_clips": stats["total"],
                        "skipped_non_video": stats["non_video"],
                        "skipped_missing_path": stats["missing_path"] + stats["path_not_found"],
                    },
                }
            }

        results: list[dict[str, Any]] = []
        failed = 0
        for item in selected:
            clip_id = item.get("clip_id", "")
            clip_path = item.get("path", "")
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    wav_path = _extract_audio(clip_path, tmp_dir, sr=sample_rate)
                    energy = compute_audio_energy(
                        wav_path,
                        sr=sample_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                    )

                times = energy.get("times") or []
                rms_db = energy.get("rms_db") or []
                low_db = energy.get("low_freq_energy_db") or []
                high_db = energy.get("high_freq_energy_db") or []

                count = min(len(times), len(rms_db), len(low_db), len(high_db))
                idx = _sample_indices(count, max_points)
                sampled_times = [float(times[i]) for i in idx]
                sampled_rms = [float(rms_db[i]) for i in idx]
                sampled_low = [float(low_db[i]) for i in idx]
                sampled_high = [float(high_db[i]) for i in idx]

                if count > 0:
                    rms_slice = np.asarray(rms_db[:count], dtype=float)
                    dynamic_range_db = float(np.percentile(rms_slice, 95) - np.percentile(rms_slice, 10))
                    rms_db_mean = float(np.mean(rms_slice))
                else:
                    dynamic_range_db = 0.0
                    rms_db_mean = 0.0

                results.append(
                    {
                        "clip_id": clip_id,
                        "path": clip_path,
                        "duration_sec": float(energy.get("duration_sec", 0.0) or 0.0),
                        "sample_rate": int(energy.get("sample_rate", sample_rate) or sample_rate),
                        "rms_db_mean": round(rms_db_mean, 3),
                        "dynamic_range_db": round(dynamic_range_db, 3),
                        "curve": {
                            "times_sec": sampled_times,
                            "rms_db": sampled_rms,
                            "low_freq_energy_db": sampled_low,
                            "high_freq_energy_db": sampled_high,
                        },
                    }
                )
            except Exception as e:
                failed += 1
                node_state.node_summary.add_warning(f"[audio-energy] {clip_id} failed: {e}")

        node_state.node_summary.info_for_user(
            f"Audio-energy analysis completed: {len(results)} clip(s) analyzed, {failed} failed."
        )
        return {
            "audio_energy": {
                "clips": results,
                "summary": {
                    "analyzed_video_clips": len(results),
                    "failed_video_clips": failed,
                    "total_clips": stats["total"],
                    "skipped_non_video": stats["non_video"],
                    "skipped_missing_path": stats["missing_path"] + stats["path_not_found"],
                },
            }
        }
