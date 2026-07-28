"""
Media Processing & Transformation Engine
========================================
Core backend for modular video/audio signal transformations.
Integrates:
  1. Traditional FFmpeg transforms (re-encode, scale, pitch, crop)
  2. Adversarial frame perturbation (gradient-optimized pixel noise)
  3. Metadata injection / compositing
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from adversarial_engine import AdversarialPerturber, DifferentiableHashExtractor


class VideoTransformationEngine:
    """
    Complete video transformation pipeline combining FFmpeg-based stream
    processing with optional adversarial frame perturbation.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_path
        self.perturber = AdversarialPerturber(
            target_hash_fn=DifferentiableHashExtractor.combined_hash,
            epsilon=8.0,
            steps=40,
            learning_rate=0.01
        )

    def transform(
        self,
        input_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full transformation pipeline.

        Config keys:
            Video: video_codec, resolution_scale, crf, frame_rate,
                   interpolation, grain_strength
            Audio: audio_codec, audio_bitrate, pitch_shift, eq_filter
            Evasion: mirror, zoom, zoom_factor, micro_rotate, speed,
                     add_border
            Adversarial: adversarial_enabled, adversarial_epsilon,
                         adversarial_steps, adversarial_batch_size
            Metadata: strip_metadata, inject_metadata
            Container: container
        """
        if config is None:
            config = self.get_default_config()

        input_file = Path(input_path).resolve()
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        adversarial_metrics = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            working_video = str(input_file)

            # ── Step 1: Adversarial Frame Perturbation (optional) ──
            if config.get("adversarial_enabled", False):
                self.perturber.epsilon = float(config.get("adversarial_epsilon", 8.0)) / 255.0
                self.perturber.steps = int(config.get("adversarial_steps", 40))

                perturbed_video = str(tmp_path / "adversarial_frames.mp4")
                adversarial_metrics = self.perturber.perturb_video(
                    working_video,
                    perturbed_video,
                    batch_size=int(config.get("adversarial_batch_size", 4))
                )
                working_video = perturbed_video

            # ── Step 2: Video Stream Transforms (FFmpeg) ──
            temp_video = str(tmp_path / "processed_video.mp4")
            self._transform_video_stream(working_video, temp_video, config)

            # ── Step 3: Audio Stream Transforms ──
            temp_audio = str(tmp_path / "processed_audio.wav")
            self._transform_audio_stream(str(input_file), temp_audio, config)

            # ── Step 4: Remux with metadata handling ──
            self._remux(temp_video, temp_audio, str(output_file), config)

        original_size = input_file.stat().st_size
        output_size = output_file.stat().st_size if output_file.exists() else 0

        result = {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "original_size_bytes": original_size,
            "output_size_bytes": output_size,
            "size_reduction_pct": round((1 - output_size / original_size) * 100, 2) if original_size > 0 else 0,
            "config": config,
            "status": "completed"
        }

        if adversarial_metrics:
            result["adversarial_metrics"] = adversarial_metrics

        return result

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            # Video
            "video_codec": "h264",
            "resolution_scale": 0.85,
            "crf": 24,
            "frame_rate": 30,
            "interpolation": "bicubic",
            "grain_strength": 0.0,
            # Audio
            "audio_codec": "aac",
            "audio_bitrate": "128k",
            "pitch_shift": 0.0,
            "eq_filter": False,
            # Evasion transforms
            "mirror": False,
            "zoom": False,
            "zoom_factor": 1.05,
            "speed": 1.0,
            "micro_rotate": False,
            "add_border": False,
            # Adversarial
            "adversarial_enabled": False,
            "adversarial_epsilon": 8.0,
            "adversarial_steps": 40,
            "adversarial_batch_size": 4,
            # Metadata
            "strip_metadata": True,
            "inject_metadata": False,
            # Container
            "container": "mp4"
        }

    def _transform_video_stream(self, input_path: str, output_path: str, config: Dict[str, Any]):
        codec_map = {
            "h264": "libx264",
            "h265": "libx265",
            "av1": "libaom-av1"
        }
        codec = codec_map.get(config.get("video_codec", "h264"), "libx264")

        vf_filters = []

        # Mirror (horizontal flip)
        if config.get("mirror", False):
            vf_filters.append("hflip")

        # Zoom-crop
        if config.get("zoom", False):
            zoom = float(config.get("zoom_factor", 1.05))
            vf_filters.append(
                f"scale=iw*{zoom}:ih*{zoom}:flags=lanczos,"
                f"crop=iw/{zoom}:ih/{zoom}"
            )

        # Resolution scaling
        scale = float(config.get("resolution_scale", 1.0))
        if scale < 1.0 and scale > 0.1:
            interpolation = config.get("interpolation", "bicubic")
            vf_filters.append(
                f"scale=w='trunc(iw*{scale}/2)*2':h='trunc(ih*{scale}/2)*2':flags={interpolation}"
            )

        # Micro-rotation
        if config.get("micro_rotate", False):
            vf_filters.append("rotate=0.3*PI/180:fillcolor=black")

        # Border overlay (compositing layer)
        if config.get("add_border", False):
            vf_filters.append("drawbox=x=0:y=0:w=iw:h=ih:color=black@0.03:t=3")

        # Grain / Noise overlay
        grain = float(config.get("grain_strength", 0.0))
        if grain > 0.0:
            noise_val = max(1, int(grain * 20))
            vf_filters.append(f"noise=alls={noise_val}:allf=t+u")

        # Speed adjustment (setpts for video)
        speed = float(config.get("speed", 1.0))
        if speed != 1.0 and speed > 0.5 and speed < 2.0:
            vf_filters.append(f"setpts=PTS/{speed}")

        cmd = [
            self.ffmpeg, "-y",
            "-i", input_path,
            "-an",
            "-c:v", codec,
            "-crf", str(config.get("crf", 24)),
            "-preset", "fast",
            "-pix_fmt", "yuv420p"
        ]

        if config.get("frame_rate"):
            cmd.extend(["-r", str(config.get("frame_rate"))])

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg Video Error: {result.stderr[:500]}")

    def _transform_audio_stream(self, input_path: str, output_path: str, config: Dict[str, Any]):
        af_filters = []

        # Pitch shift via asetrate + atempo (more compatible than rubberband)
        pitch_shift = float(config.get("pitch_shift", 0.0))
        if pitch_shift != 0.0:
            multiplier = 2.0 ** (pitch_shift / 12.0)
            new_rate = int(44100 * multiplier)
            tempo_correction = 1.0 / multiplier
            af_filters.append(f"asetrate={new_rate},atempo={tempo_correction:.6f}")

        # Speed adjustment for audio (match video speed)
        speed = float(config.get("speed", 1.0))
        if speed != 1.0 and speed > 0.5 and speed < 2.0:
            af_filters.append(f"atempo={speed}")

        # EQ filter
        if config.get("eq_filter", False):
            af_filters.append(
                "equalizer=f=1000:t=q:w=1:g=-4,"
                "equalizer=f=3000:t=q:w=1:g=-3"
            )

        cmd = [
            self.ffmpeg, "-y",
            "-i", input_path,
            "-vn",
            "-ar", "44100",
            "-ac", "2"
        ]

        if af_filters:
            cmd.extend(["-af", ",".join(af_filters)])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback: extract audio without filters
            cmd_fallback = [
                self.ffmpeg, "-y", "-i", input_path,
                "-vn", "-ar", "44100", "-ac", "2",
                output_path
            ]
            fallback = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if fallback.returncode != 0:
                # Video may have no audio — create silent audio
                cmd_silent = [
                    self.ffmpeg, "-y",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", "1",
                    output_path
                ]
                subprocess.run(cmd_silent, capture_output=True)

    def _remux(self, video_path: str, audio_path: str, output_path: str, config: Dict[str, Any]):
        codec_map = {
            "aac": "aac",
            "mp3": "libmp3lame",
            "opus": "libopus"
        }
        audio_codec = codec_map.get(config.get("audio_codec", "aac"), "aac")

        cmd = [
            self.ffmpeg, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", audio_codec,
            "-b:a", config.get("audio_bitrate", "128k"),
            "-shortest"
        ]

        # Metadata handling
        if config.get("inject_metadata", False):
            cmd.extend([
                "-metadata", "creation_time=2026-07-27T14:30:00Z",
                "-metadata", "model=iPhone15,3",
                "-metadata", "make=Apple",
                "-metadata", "encoder=VideoToolBox"
            ])
        elif config.get("strip_metadata", True):
            cmd.extend(["-map_metadata", "-1"])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg Remux Error: {result.stderr[:500]}")
