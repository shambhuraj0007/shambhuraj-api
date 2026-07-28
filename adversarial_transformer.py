#!/usr/bin/env python3
"""
Adversarial Video Transformation CLI
=====================================
Standalone command-line tool that runs the full adversarial transformation
pipeline against a video file, with optional benchmarking mode.

Usage:
    python adversarial_transformer.py input.mp4 output.mp4
    python adversarial_transformer.py --benchmark input.mp4
"""

import os
import sys
import json
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from typing import Callable, Optional, Dict, List

from adversarial_engine import AdversarialPerturber, DifferentiableHashExtractor


class FullTransformationPipeline:
    """
    Complete video transformation engine combining:
      1. Adversarial frame perturbation (targeting your detection algorithm)
      2. Traditional evasion transforms (mirror, zoom-crop, speed, re-encode)
      3. Audio processing (pitch shift, EQ, compression)
      4. Metadata injection

    This is the standalone CLI version. For the web UI, see benchmark.py + app.py.
    """

    def __init__(
        self,
        target_hash_fn: Optional[Callable] = None,
        ffmpeg_path: str = "ffmpeg"
    ):
        if target_hash_fn is None:
            target_hash_fn = DifferentiableHashExtractor.combined_hash

        self.perturber = AdversarialPerturber(
            target_hash_fn=target_hash_fn,
            epsilon=8.0,
            steps=50,
            learning_rate=0.01
        )
        self.ffmpeg = ffmpeg_path

    def transform(
        self,
        input_path: str,
        output_path: str,
        config: Optional[Dict] = None
    ) -> Dict:
        """
        Full transformation pipeline.

        Returns dict with results and adversarial quality metrics.
        """
        if config is None:
            config = self._default_config()

        results = {
            "input": input_path,
            "output": output_path,
            "config": config,
            "steps": {}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # ── Step 1: Extract audio ──
            print("\n=== Step 1: Extract Audio ===")
            audio_extracted = tmp / "audio_original.wav"
            r = subprocess.run([
                self.ffmpeg, "-i", input_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2",
                "-y", str(audio_extracted)
            ], capture_output=True)
            has_audio = audio_extracted.exists() and audio_extracted.stat().st_size > 1000

            # ── Step 2: Adversarial frame perturbation ──
            print("\n=== Step 2: Adversarial Frame Perturbation ===")
            raw_video = tmp / "raw_frames.mp4"
            subprocess.run([
                self.ffmpeg, "-i", input_path,
                "-an", "-c:v", "libx264", "-crf", "0",
                "-preset", "fast",
                "-y", str(raw_video)
            ], capture_output=True, check=True)

            perturbed_video = tmp / "perturbed.mp4"
            perturb_metrics = self.perturber.perturb_video(
                str(raw_video), str(perturbed_video),
                batch_size=config.get("batch_size", 4)
            )
            results["steps"]["adversarial_perturbation"] = perturb_metrics

            # ── Step 3: Traditional video transforms ──
            print("\n=== Step 3: Traditional Evasion Transforms ===")
            transformed_video = tmp / "transformed.mp4"
            self._apply_traditional_transforms(
                str(perturbed_video), str(transformed_video), config, has_audio
            )

            # ── Step 4: Audio transformation ──
            print("\n=== Step 4: Audio Transformation ===")
            if has_audio:
                processed_audio = tmp / "audio_processed.wav"
                self._transform_audio(str(audio_extracted), str(processed_audio), config)
                final_audio = str(processed_audio)
            else:
                final_audio = None

            # ── Step 5: Remux with metadata injection ──
            print("\n=== Step 5: Remux + Metadata Injection ===")
            self._remux_with_metadata(
                str(transformed_video), final_audio, output_path, config
            )

            if os.path.getsize(output_path) < 1000:
                raise RuntimeError("Output file is too small — transformation failed")

            results["status"] = "completed"
            results["output_size_mb"] = round(os.path.getsize(output_path) / (1024 * 1024), 2)

        print(f"\n{'=' * 50}")
        print(f"Transformation complete!")
        print(f"Output: {output_path} ({results['output_size_mb']} MB)")
        print(f"PSNR:      {perturb_metrics['avg_psnr_db']:.2f} dB")
        print(f"Hash Sim:  {perturb_metrics['avg_cosine_similarity']:.4f}")
        print(f"{'=' * 50}")

        return results

    def _apply_traditional_transforms(self, input_path, output_path, config, has_audio):
        vf_parts = []

        if config.get("mirror", True):
            vf_parts.append("hflip")

        if config.get("zoom", True):
            zoom = config.get("zoom_factor", 1.05)
            vf_parts.append(
                f"scale=iw*{zoom}:ih*{zoom}:flags=lanczos,"
                f"crop=iw/{zoom}:ih/{zoom}"
            )

        speed = config.get("speed", 1.04)
        if speed != 1.0:
            vf_parts.append(f"setpts=PTS/{speed}")

        if config.get("add_border", True):
            vf_parts.append("drawbox=x=0:y=0:w=iw:h=ih:color=black@0.02:t=2")

        if config.get("micro_rotate", True):
            vf_parts.append("rotate=0.3*PI/180:fillcolor=black")

        vf_str = ",".join(vf_parts) if vf_parts else None

        cmd = [self.ffmpeg, "-i", input_path]
        cmd.extend(["-an"] if not has_audio else ["-c:a", "copy"])
        if vf_str:
            cmd.extend(["-vf", vf_str])

        cmd.extend([
            "-c:v", config.get("video_codec", "libx265"),
            "-crf", str(config.get("crf", 26)),
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-y", output_path
        ])
        subprocess.run(cmd, capture_output=True, check=True)

    def _transform_audio(self, input_audio, output_audio, config):
        af_parts = []

        semitones = config.get("pitch_shift", -2.0)
        if semitones != 0:
            multiplier = 2.0 ** (semitones / 12.0)
            new_rate = int(44100 * multiplier)
            tempo_corr = 1.0 / multiplier
            af_parts.append(f"asetrate={new_rate},atempo={tempo_corr:.6f}")

        if config.get("eq_filter"):
            af_parts.append(config["eq_filter"])

        if config.get("compression", True):
            af_parts.append(
                "compand=attacks=0.1:decays=0.1:"
                "points=-80/-80|-30/-30|-16/-10|0/-3|20/-3"
            )

        af_str = ",".join(af_parts) if af_parts else "anull"

        codec_map = {
            "aac": "aac", "mp3": "libmp3lame",
            "opus": "libopus", "vorbis": "libvorbis"
        }
        audio_codec = codec_map.get(config.get("audio_codec", "opus"), "libopus")

        subprocess.run([
            self.ffmpeg, "-i", input_audio,
            "-af", af_str,
            "-c:a", audio_codec,
            "-b:a", config.get("audio_bitrate", "96k"),
            "-y", output_audio
        ], capture_output=True, check=True)

    def _remux_with_metadata(self, video_path, audio_path, output_path, config):
        cmd = [self.ffmpeg]

        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-i", video_path, "-i", audio_path,
                        "-map", "0:v", "-map", "1:a"])
        else:
            cmd.extend(["-i", video_path])

        cmd.extend(["-c:v", "copy", "-c:a", "copy"])

        if config.get("inject_metadata", True):
            cmd.extend([
                "-metadata", "creation_time=2026-07-27T14:30:00Z",
                "-metadata", "model=iPhone15,3",
                "-metadata", "make=Apple",
                "-metadata", "encoder=VideoToolBox"
            ])

        cmd.extend(["-y", output_path])
        subprocess.run(cmd, capture_output=True, check=True)

    @staticmethod
    def _default_config() -> Dict:
        return {
            "video_codec": "libx265",
            "crf": 26,
            "speed": 1.04,
            "mirror": True,
            "zoom": True,
            "zoom_factor": 1.05,
            "micro_rotate": True,
            "add_border": True,
            "pitch_shift": -2.0,
            "eq_filter": "equalizer=f=3000:t=q:w=1:g=-6",
            "compression": True,
            "audio_codec": "opus",
            "audio_bitrate": "96k",
            "inject_metadata": True,
            "batch_size": 4
        }


class AlgorithmBenchmarker:
    """
    Tests a detection algorithm against transformed video variants.
    Measures detection rate across escalating transformation intensity.
    """

    def __init__(self, detection_algorithm: Callable[[str], Dict]):
        """
        detection_algorithm: Function (video_path) -> {'detected': bool, 'confidence': float, ...}
        """
        self.detect = detection_algorithm

    def benchmark(
        self,
        video_path: str,
        pipeline: FullTransformationPipeline,
        output_dir: str = "benchmark_outputs"
    ) -> List[Dict]:
        """Run multiple transformation configs and test detection against each."""
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(video_path).stem
        results = []

        # Test 1: Baseline — unmodified video
        print("\n" + "=" * 50)
        print("TEST 1: Baseline (no transformation)")
        print("=" * 50)
        baseline = self.detect(video_path)
        results.append({
            "config": "baseline (no transform)",
            "detected": baseline.get("detected", False),
            "confidence": baseline.get("confidence", 0)
        })

        # Test 2: Traditional transforms only
        print("\n" + "=" * 50)
        print("TEST 2: Traditional transforms only")
        print("=" * 50)
        trad_output = os.path.join(output_dir, f"{stem}_traditional.mp4")
        trad_config = pipeline._default_config()
        # Skip adversarial step by setting steps=0
        orig_steps = pipeline.perturber.steps
        pipeline.perturber.steps = 0
        try:
            pipeline.transform(video_path, trad_output, trad_config)
        except Exception:
            # If 0-step perturbation fails, just copy through
            pass
        pipeline.perturber.steps = orig_steps
        if os.path.exists(trad_output):
            trad_result = self.detect(trad_output)
            results.append({
                "config": "traditional only",
                "detected": trad_result.get("detected", False),
                "confidence": trad_result.get("confidence", 0)
            })

        # Test 3: Full adversarial pipeline (eps=8)
        print("\n" + "=" * 50)
        print("TEST 3: Full adversarial (eps=8/255)")
        print("=" * 50)
        adv_output = os.path.join(output_dir, f"{stem}_adversarial_eps8.mp4")
        pipeline.perturber.epsilon = 8.0 / 255.0
        pipeline.perturber.steps = 50
        pipeline.transform(video_path, adv_output, pipeline._default_config())
        adv_result = self.detect(adv_output)
        results.append({
            "config": "full adversarial (eps=8/255)",
            "detected": adv_result.get("detected", False),
            "confidence": adv_result.get("confidence", 0)
        })

        # Test 4-5: Epsilon sweep
        for eps in [4, 12]:
            print(f"\n{'=' * 50}")
            print(f"TEST: Epsilon sweep (eps={eps}/255)")
            print(f"{'=' * 50}")
            eps_output = os.path.join(output_dir, f"{stem}_adversarial_eps{eps}.mp4")
            pipeline.perturber.epsilon = eps / 255.0
            pipeline.transform(video_path, eps_output, pipeline._default_config())
            eps_result = self.detect(eps_output)
            results.append({
                "config": f"adversarial (eps={eps}/255)",
                "detected": eps_result.get("detected", False),
                "confidence": eps_result.get("confidence", 0)
            })

        return results


def demo_detection_algorithm(video_path: str) -> Dict:
    """
    Placeholder detection algorithm for demonstration.
    Replace this with YOUR actual copyright detection logic.

    Your function should:
      1. Load the video
      2. Compute your perceptual hash / deep features
      3. Compare against a reference database or threshold
      4. Return detection results
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"detected": False, "confidence": 0.0, "error": "could not read"}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    hash_val = DifferentiableHashExtractor.combined_hash(tensor)

    # Compare against a baseline "known" hash (placeholder)
    known_hash = torch.ones_like(hash_val) * 0.5
    similarity = F.cosine_similarity(hash_val, known_hash).item()

    return {
        "detected": similarity > 0.8,
        "confidence": round(float(similarity), 4)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("VORTEX Adversarial Video Transformer CLI")
        print()
        print("Usage:")
        print("  Transform:  python adversarial_transformer.py input.mp4 output.mp4")
        print("  Benchmark:  python adversarial_transformer.py --benchmark input.mp4")
        sys.exit(1)

    if sys.argv[1] == "--benchmark":
        video_path = sys.argv[2]

        pipeline = FullTransformationPipeline(
            target_hash_fn=DifferentiableHashExtractor.combined_hash
        )

        benchmarker = AlgorithmBenchmarker(
            detection_algorithm=demo_detection_algorithm
        )
        results = benchmarker.benchmark(video_path, pipeline)

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        for r in results:
            status = "EVADED" if not r["detected"] else "DETECTED"
            print(f"  {r['config']:40s} -> {status}  (conf={r['confidence']:.3f})")
        print("=" * 60)

        # Save results
        results_path = "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output_evaded.mp4"

        pipeline = FullTransformationPipeline(
            target_hash_fn=DifferentiableHashExtractor.combined_hash
        )

        results = pipeline.transform(input_path, output_path)

        # Save metrics
        metrics_path = output_path + ".metrics.json"

        def convert(o):
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"Metrics saved to: {metrics_path}")
