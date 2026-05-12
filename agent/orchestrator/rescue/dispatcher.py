"""Rescue mission dispatcher: bridges the vision thread to the asyncio event loop.

When the person-rescue trigger fires on the camera thread, it calls
``RescueMissionDispatcher.request_rescue``. That method submits a coroutine to the
asyncio loop via ``run_coroutine_threadsafe`` so the camera thread is never blocked.

The coroutine uploads a deterministic return-home-and-land mission via gRPC, then
optionally starts a parallel asyncio task that asks Gemma to analyse the saved crop
image (when ``image_llm_enabled`` is true) and log a situation estimate and action plan.
"""

import asyncio
import logging
import math
import time
from pathlib import Path
from typing import Any

from agent.orchestrator.grpc_client import InternalGrpcClient
from agent.orchestrator.llm.client import LlamaClient
from agent.orchestrator.llm.prompts import (
    build_rescue_analysis_system_prompt,
    build_rescue_analysis_user_prompt,
)
from agent.orchestrator.logging import JsonPipelineLogger, log_mission_multipoint_geojson
from agent.orchestrator.mission_intents import expand_intents_to_mission
from agent.orchestrator.mission_intents.proto import mission_list_to_ordered_dict
from agent.orchestrator.rescue.geo import (
    PersonGeoEstimate,
    PersonOffset,
    estimate_person_lat_lon,
    estimate_person_offset,
)
from agent.orchestrator.rescue.home_state import HomeLocationState
from agent.orchestrator.state import TelemetryCache

log = logging.getLogger(__name__)


def _telemetry_float(tel_map: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(tel_map.get(key, default))
    except (TypeError, ValueError):
        return default


def _rename_rescue_snapshots(
    *,
    full_path: Path,
    crop_path: Path | None,
    geo: PersonGeoEstimate,
    decimals: int = 6,
) -> tuple[Path, Path | None]:
    """Rename JPEGs to ``{lat}?{lon}_full.jpg`` / ``_person.jpg``; on collision append a suffix."""
    stem = f"{geo.latitude_deg:.{decimals}f}?{geo.longitude_deg:.{decimals}f}"

    def _unique(dest: Path) -> Path:
        if not dest.exists():
            return dest
        return dest.parent / f"{stem}_{int(time.time() * 1000)}{dest.suffix}"

    new_full = _unique(full_path.parent / f"{stem}_full.jpg")
    full_path.rename(new_full)
    new_crop: Path | None = None
    if crop_path is not None and crop_path.is_file():
        new_crop = _unique(crop_path.parent / f"{stem}_person.jpg")
        crop_path.rename(new_crop)
    return new_full, new_crop


class RescueMissionDispatcher:
    """Bridges the vision thread to asyncio for rescue mission upload and Gemma analysis.

    Constructed once inside the asyncio event loop after a live gRPC client exists.
    The public method ``request_rescue`` is safe to call from any thread.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        client: InternalGrpcClient,
        llm: LlamaClient,
        cache: TelemetryCache,
        home_state: HomeLocationState,
        json_logger: JsonPipelineLogger,
        min_rth_alt_m: float,
        camera_mount_pitch_deg: float,
        camera_hfov_deg: float,
        camera_vfov_deg: float,
        image_llm_enabled: bool = True,
    ) -> None:
        """Initialise the dispatcher.

        Args:
            loop: The running asyncio event loop. Used by ``run_coroutine_threadsafe``.
            client: Live gRPC client for uploading missions.
            llm: LlamaClient connected to llama-server (use ``--mmproj`` when
                ``image_llm_enabled`` is true for rescue image analysis).
            cache: Shared telemetry cache; queried at trigger time for current altitude.
            home_state: Stores the first-takeoff lat/lon used as the RTH destination.
            json_logger: Pipeline JSONL logger for all rescue events.
            min_rth_alt_m: Safety floor for the RTH cruise altitude in metres. The drone
                will never fly home below this altitude even if current telemetry is lower.
            camera_mount_pitch_deg: Camera tilt from horizontal in degrees (90 = nadir).
            camera_hfov_deg: Camera horizontal field of view in degrees.
            camera_vfov_deg: Camera vertical field of view in degrees.
            image_llm_enabled: If false, return-home upload still runs but the follow-up
                multimodal crop analysis is skipped (text-only ``plan_mission`` remains available).
        """
        self._loop = loop
        self._client = client
        self._llm = llm
        self._cache = cache
        self._home_state = home_state
        self._json_logger = json_logger
        self._min_rth_alt_m = float(min_rth_alt_m)
        self._camera_mount_pitch_deg = float(camera_mount_pitch_deg)
        self._camera_hfov_deg = float(camera_hfov_deg)
        self._camera_vfov_deg = float(camera_vfov_deg)
        self._image_llm_enabled = bool(image_llm_enabled)
        # Serialises Gemma calls so mission planning and analysis do not interleave.
        self._llm_lock = asyncio.Lock()

    def request_rescue(
        self,
        *,
        bbox_xyxy: tuple[float, float, float, float],
        image_wh: tuple[int, int],
        full_path: Path,
        crop_path: Path | None,
    ) -> None:
        """Submit a rescue coroutine from any thread.

        Returns immediately; the actual gRPC upload and Gemma analysis run on the
        asyncio event loop without blocking the caller.

        Args:
            bbox_xyxy: Bounding box of the detected person in preview-frame pixel
                coordinates (x1, y1, x2, y2). Used for the body-frame offset estimate.
            image_wh: Width and height of the preview frame in pixels.
            full_path: Path to the annotated full-frame JPEG (may be renamed after geo estimate).
            crop_path: Path to the saved person-crop JPEG for multimodal analysis, or
                ``None`` when only a full-frame snapshot was saved.
        """
        coro = self._run_rescue(
            bbox_xyxy=bbox_xyxy,
            image_wh=image_wh,
            full_path=full_path,
            crop_path=crop_path,
        )
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _run_rescue(
        self,
        *,
        bbox_xyxy: tuple[float, float, float, float],
        image_wh: tuple[int, int],
        full_path: Path,
        crop_path: Path | None,
    ) -> None:
        """Upload return-home mission then start the Gemma analysis task.

        Args:
            bbox_xyxy: Person bounding box in preview pixels (x1, y1, x2, y2).
            image_wh: Preview frame dimensions (width, height) in pixels.
            full_path: Annotated full-frame JPEG path (renamed to estimated lat?lon when valid).
            crop_path: Path to the person-crop JPEG for the Gemma multimodal call, or
                ``None`` if no crop was saved.
        """
        home = self._home_state.get()
        if home is None:
            log.warning("Rescue trigger fired but home location is not set; skipping mission.")
            return

        tel_map: dict[str, Any] = await self._cache.as_any()
        trace_id = self._json_logger.new_trace_id()

        # relative_altitude_m is AGL as reported by the flight controller.
        # Clamped to min_rth_alt_m so the drone never descends to fly home.
        rel_alt = _telemetry_float(tel_map, "relative_altitude_m")
        if math.isnan(rel_alt):
            rth_alt = self._min_rth_alt_m
        else:
            rth_alt = max(rel_alt, self._min_rth_alt_m)

        lat_deg = _telemetry_float(tel_map, "latitude_deg")
        lon_deg = _telemetry_float(tel_map, "longitude_deg")
        yaw_deg = _telemetry_float(tel_map, "yaw_deg")

        offset: PersonOffset
        geo: PersonGeoEstimate | None
        rel_alt_geom = rel_alt if not math.isnan(rel_alt) else 0.0

        if not any(math.isnan(x) for x in (lat_deg, lon_deg, yaw_deg, rel_alt)):
            offset, geo_est = estimate_person_lat_lon(
                bbox_xyxy,
                image_wh,
                drone_latitude_deg=lat_deg,
                drone_longitude_deg=lon_deg,
                drone_altitude_m=rel_alt,
                yaw_deg=yaw_deg,
                camera_pitch_deg=self._camera_mount_pitch_deg,
                hfov_deg=self._camera_hfov_deg,
                vfov_deg=self._camera_vfov_deg,
            )
            geo = geo_est
            self._json_logger.log(
                "rescue_person_geo_estimated",
                trace_id,
                {
                    "person_latitude_deg": geo.latitude_deg,
                    "person_longitude_deg": geo.longitude_deg,
                    "north_offset_m": geo.north_offset_m,
                    "east_offset_m": geo.east_offset_m,
                    "forward_m": offset.forward_m,
                    "right_m": offset.right_m,
                    "drone_latitude_deg": lat_deg,
                    "drone_longitude_deg": lon_deg,
                    "yaw_deg": yaw_deg,
                    "relative_altitude_m": rel_alt,
                },
            )
            log.info(
                "Rescue person geo estimate: person=(%.6f, %.6f) drone=(%.6f, %.6f) "
                "yaw_deg=%.1f AGL=%.1fm N=%.2fm E=%.2fm forward=%.2fm right=%.2fm",
                geo.latitude_deg,
                geo.longitude_deg,
                lat_deg,
                lon_deg,
                yaw_deg,
                rel_alt,
                geo.north_offset_m,
                geo.east_offset_m,
                offset.forward_m,
                offset.right_m,
            )
            try:
                full_path, crop_path = _rename_rescue_snapshots(
                    full_path=full_path, crop_path=crop_path, geo=geo
                )
                log.info(
                    "Rescue snapshots renamed to lat?lon stem: full=%s crop=%s",
                    full_path,
                    crop_path if crop_path is not None else "(none)",
                )
            except OSError as exc:
                log.warning("Could not rename rescue snapshots to geo stem: %s", exc)
        else:
            geo = None
            offset = estimate_person_offset(
                bbox_xyxy=bbox_xyxy,
                image_wh=image_wh,
                drone_altitude_m=max(rel_alt_geom, 0.0),
                camera_pitch_deg=self._camera_mount_pitch_deg,
                hfov_deg=self._camera_hfov_deg,
                vfov_deg=self._camera_vfov_deg,
            )
            log.warning(
                "Rescue geo estimate skipped (invalid telemetry); snapshots keep timestamp names. "
                "tel_map keys lat/lon/yaw/rel_alt may be NaN or missing."
            )
        plan = self._build_return_home_plan(
            home_lat=home.latitude_deg,
            home_lon=home.longitude_deg,
            rth_alt=rth_alt,
        )
        self._json_logger.log("rescue_plan_built", trace_id, {"mission_plan": plan})

        try:
            proto = expand_intents_to_mission(plan, tel_map)
        except Exception as exc:
            log.exception("expand_intents_to_mission for rescue failed: %s", exc)
            self._json_logger.log(
                "rescue_mission_failed",
                trace_id,
                {"stage": "intent_expansion", "error": str(exc), "mission_plan": plan},
            )
            return

        self._json_logger.log(
            "rescue_mission_converted",
            trace_id,
            {"mission_plan": plan, "mission_proto": mission_list_to_ordered_dict(proto)},
        )
        log_mission_multipoint_geojson(self._json_logger, trace_id, proto)

        try:
            await self._client.start_mission(proto)
        except Exception as exc:
            log.exception("StartMission for rescue failed: %s", exc)
            self._json_logger.log(
                "rescue_mission_failed",
                trace_id,
                {"stage": "start_mission_rpc", "error": str(exc)},
            )
            return

        self._json_logger.log(
            "rescue_mission_uploaded",
            trace_id,
            {"mission_name": plan.get("mission_name"), "item_count": len(proto.items)},
        )

        if not self._image_llm_enabled:
            log.info(
                "Rescue image LLM analysis disabled (RESCUE_IMAGE_LLM_ENABLED); "
                "skipping multimodal crop prompt."
            )
            self._json_logger.log(
                "rescue_analysis_skipped",
                trace_id,
                {"reason": "image_llm_disabled"},
            )
            return

        # Fire-and-forget: analysis runs while the drone is already flying home.
        asyncio.create_task(
            self._run_analysis(
                trace_id=trace_id,
                tel_map=tel_map,
                crop_path=crop_path,
                person_offset=offset,
                person_geo=geo,
            ),
            name="rescue_analysis",
        )

    def _build_return_home_plan(
        self,
        *,
        home_lat: float,
        home_lon: float,
        rth_alt: float,
    ) -> dict[str, Any]:
        """Build a deterministic return-home-and-land mission intent plan.

        The plan uses ``goto_lat_lon`` with the stored first-takeoff coordinates rather
        than ``return_to_home``, because ``return_to_home`` is a relative reversal of the
        current mission's cumulative offset and would fly to the wrong place when triggered
        mid-mission.

        Args:
            home_lat: Latitude of the first-takeoff point in decimal degrees.
            home_lon: Longitude of the first-takeoff point in decimal degrees.
            rth_alt: Cruise altitude for the return leg in metres AGL.

        Returns:
            A mission plan dict ready for ``expand_intents_to_mission``.
        """
        return {
            "mission_name": "rescue_rth",
            "intents": [
                {
                    "type": "goto_lat_lon",
                    "latitude_deg": float(home_lat),
                    "longitude_deg": float(home_lon),
                    "altitude_m": float(rth_alt),
                },
                {"type": "land"},
            ],
        }

    async def _run_analysis(
        self,
        *,
        trace_id: str,
        tel_map: dict[str, Any],
        crop_path: Path | None,
        person_offset: PersonOffset,
        person_geo: PersonGeoEstimate | None,
    ) -> None:
        """Send the person crop to Gemma and log the situation analysis.

        Runs as a separate asyncio task so it does not delay the mission upload.
        Acquires ``_llm_lock`` to prevent interleaving with normal mission-planning calls.

        Args:
            trace_id: Pipeline trace ID to correlate all rescue log events.
            tel_map: Telemetry snapshot taken at trigger time.
            crop_path: Path to the person-crop JPEG to send to Gemma.
            person_offset: Body-frame offset computed alongside geo (or geometry-only if no tel).
            person_geo: Estimated person WGS84 position when telemetry was valid, else ``None``.
        """
        if crop_path is None:
            log.warning("Rescue analysis skipped: no crop_path (unexpected with image LLM enabled).")
            self._json_logger.log(
                "rescue_analysis_failed",
                trace_id,
                {"error": "missing_crop_path"},
            )
            return
        rel_alt = _telemetry_float(tel_map, "relative_altitude_m")
        lat_deg = _telemetry_float(tel_map, "latitude_deg")
        lon_deg = _telemetry_float(tel_map, "longitude_deg")

        offset = person_offset
        offset_payload: dict[str, Any] = {
            "latitude_deg": lat_deg,
            "longitude_deg": lon_deg,
            "forward_m": offset.forward_m,
            "right_m": offset.right_m,
            "relative_altitude_m": rel_alt,
        }
        if person_geo is not None:
            offset_payload["person_latitude_deg"] = person_geo.latitude_deg
            offset_payload["person_longitude_deg"] = person_geo.longitude_deg
        self._json_logger.log("rescue_person_offset_estimated", trace_id, offset_payload)

        try:
            image_bytes = crop_path.read_bytes()
        except Exception as exc:
            log.exception("Failed to read person crop for Gemma analysis: %s", exc)
            return

        system = build_rescue_analysis_system_prompt()
        user = build_rescue_analysis_user_prompt(
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            forward_m=offset.forward_m,
            right_m=offset.right_m,
            drone_alt_m=rel_alt,
            person_latitude_deg=person_geo.latitude_deg if person_geo is not None else None,
            person_longitude_deg=person_geo.longitude_deg if person_geo is not None else None,
        )

        async with self._llm_lock:
            try:
                analysis = await self._llm.analyze_image(
                    system=system,
                    user_text=user,
                    image_jpeg=image_bytes,
                )
            except Exception as exc:
                log.exception("Rescue Gemma analysis failed: %s", exc)
                self._json_logger.log(
                    "rescue_analysis_failed", trace_id, {"error": str(exc)}
                )
                return

        log.info("Rescue analysis from Gemma:\n%s", analysis)
        done_payload: dict[str, Any] = {
            "analysis_text": analysis,
            "latitude_deg": lat_deg,
            "longitude_deg": lon_deg,
            "forward_m": offset.forward_m,
            "right_m": offset.right_m,
            "relative_altitude_m": rel_alt,
        }
        if person_geo is not None:
            done_payload["person_latitude_deg"] = person_geo.latitude_deg
            done_payload["person_longitude_deg"] = person_geo.longitude_deg
        self._json_logger.log("rescue_analysis_completed", trace_id, done_payload)
