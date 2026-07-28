"""
Adversarial Perturbation Engine
===============================
Gradient-optimized frame perturbation module for testing the robustness
of perceptual hashing and deep feature detection algorithms.

Based on techniques from USENIX Security '22 and UMD adversarial robustness
research. Implements differentiable approximations of standard perceptual
hash functions to enable gradient flow for adversarial optimization.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Callable, Optional, Dict, Tuple


class DifferentiableHashExtractor:
    """
    Differentiable approximations of perceptual hash functions.
    Enables gradient-based adversarial optimization against hash-based
    detection systems.

    Replace these with YOUR algorithm's hash function for targeted testing.
    """

    @staticmethod
    def dct_phash(frame: torch.Tensor, hash_size: int = 16) -> torch.Tensor:
        """
        Differentiable DCT-based perceptual hash (pHash).

        Args:
            frame: (B, C, H, W) tensor normalized to [0, 1]
            hash_size: Output hash dimension (hash_size x hash_size bits)

        Returns:
            (B, hash_size^2) soft hash bits in [0, 1]
        """
        B, C, H, W = frame.shape

        # Luma conversion (Rec. 601)
        gray = (0.299 * frame[:, 0:1, :, :]
                + 0.587 * frame[:, 1:2, :, :]
                + 0.114 * frame[:, 2:3, :, :])

        # Resize to (hash_size*2) x (hash_size*2) for DCT input
        dct_size = hash_size * 2
        resized = F.interpolate(
            gray, size=(dct_size, dct_size),
            mode='bilinear', align_corners=False
        )

        # 2D DCT via matrix multiplication (fully differentiable)
        dct_mat = DifferentiableHashExtractor._build_dct_matrix(dct_size, frame.device)
        dct_result = dct_mat @ resized.squeeze(1) @ dct_mat.T  # (B, N, N)

        # Extract low-frequency block (top-left hash_size x hash_size)
        low_freq = dct_result[:, :hash_size, :hash_size]

        # Soft binarization: sigmoid(temperature * (x - median))
        flat = low_freq.reshape(B, -1)
        median = flat.median(dim=1, keepdim=True)[0]
        hash_bits = torch.sigmoid(10.0 * (flat - median))

        return hash_bits  # (B, hash_size^2)

    @staticmethod
    def _build_dct_matrix(n: int, device: torch.device) -> torch.Tensor:
        """Generate n x n DCT-II matrix."""
        i, j = torch.meshgrid(
            torch.arange(n, device=device, dtype=torch.float32),
            torch.arange(n, device=device, dtype=torch.float32),
            indexing='ij'
        )
        matrix = torch.cos((j + 0.5) * np.pi * i / n) * np.sqrt(2.0 / n)
        matrix[0] *= 1.0 / np.sqrt(2)
        return matrix

    @staticmethod
    def block_hash(frame: torch.Tensor, hash_size: int = 16) -> torch.Tensor:
        """
        Differentiable block hash (difference hash variant).
        Compares mean brightness of adjacent spatial blocks.

        Args:
            frame: (B, C, H, W) tensor in [0, 1]
            hash_size: Grid resolution

        Returns:
            (B, D) soft hash bits
        """
        B, C, H, W = frame.shape
        gray = (0.299 * frame[:, 0:1, :, :]
                + 0.587 * frame[:, 1:2, :, :]
                + 0.114 * frame[:, 2:3, :, :])

        # Average pool into hash_size x hash_size grid
        pooled = F.adaptive_avg_pool2d(gray, (hash_size, hash_size))  # (B, 1, hs, hs)

        # Horizontal gradient comparison (dHash principle)
        diffs = pooled[:, :, :, 1:] - pooled[:, :, :, :-1]  # (B, 1, hs, hs-1)
        hash_bits = torch.sigmoid(10.0 * diffs)

        return hash_bits.reshape(B, -1)

    @staticmethod
    def combined_hash(frame: torch.Tensor) -> torch.Tensor:
        """
        Multi-hash concatenation for broader robustness coverage.
        Combines DCT pHash + block dHash.
        """
        phash = DifferentiableHashExtractor.dct_phash(frame, hash_size=16)
        bhash = DifferentiableHashExtractor.block_hash(frame, hash_size=12)
        return torch.cat([phash, bhash], dim=1)


class AdversarialPerturber:
    """
    Applies gradient-optimized perturbations to video frames so that the
    resulting perceptual hash diverges maximally from the original, while
    keeping the visual distortion imperceptible (within epsilon bound).
    """

    def __init__(
        self,
        target_hash_fn: Optional[Callable] = None,
        epsilon: float = 8.0,
        steps: int = 40,
        learning_rate: float = 0.01
    ):
        """
        Args:
            target_hash_fn: Differentiable hash function.
                            Signature: (B, C, H, W) -> (B, D), all in [0, 1].
                            Defaults to DifferentiableHashExtractor.combined_hash.
            epsilon: Max per-pixel perturbation magnitude (in 0-255 scale).
                     8/255 ≈ 0.031 — visually imperceptible.
            steps: Number of PGD optimization steps per batch.
            learning_rate: Adam optimizer learning rate.
        """
        self.hash_fn = target_hash_fn or DifferentiableHashExtractor.combined_hash
        self.epsilon = epsilon / 255.0  # Convert to [0, 1] scale
        self.steps = steps
        self.lr = learning_rate
        self.device = torch.device("cpu")  # CPU-only for this environment

    def perturb_batch(
        self,
        frames: np.ndarray,
        original_hashes: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply adversarial perturbation to a batch of frames.

        Args:
            frames: (B, H, W, C) uint8 array [0, 255]
            original_hashes: Pre-computed reference hashes. If None, computed internally.

        Returns:
            Tuple of (perturbed frames as uint8 ndarray, metrics dict)
        """
        B, H, W, C = frames.shape

        # Convert to (B, C, H, W) float tensor in [0, 1]
        clean = torch.from_numpy(frames).float().to(self.device) / 255.0
        clean = clean.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Compute original hash if not provided
        if original_hashes is None:
            with torch.no_grad():
                original_hashes = self.hash_fn(clean).detach()

        # Initialize perturbation delta as optimizable parameter
        delta = torch.zeros_like(clean, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        best_loss = float('inf')
        best_delta = delta.data.clone()

        for step in range(self.steps):
            optimizer.zero_grad()

            perturbed = (clean + delta).clamp(0.0, 1.0)
            current_hash = self.hash_fn(perturbed)

            # Loss: maximize hash distance (minimize cosine similarity)
            cos_sim = F.cosine_similarity(current_hash, original_hashes, dim=1)
            loss = cos_sim.mean()

            loss.backward()
            optimizer.step()

            # Project delta onto L∞ epsilon ball
            with torch.no_grad():
                delta.data.clamp_(-self.epsilon, self.epsilon)
                # Also ensure perturbed image stays valid
                delta.data = (clean + delta.data).clamp(0.0, 1.0) - clean

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.data.clone()

        # Apply best perturbation
        with torch.no_grad():
            final = (clean + best_delta).clamp(0.0, 1.0)
            final_hash = self.hash_fn(final)
            final_cos_sim = F.cosine_similarity(final_hash, original_hashes, dim=1).mean().item()

        # Convert back to numpy uint8
        result = final.permute(0, 2, 3, 1).cpu().numpy()
        result = (result * 255.0).clip(0, 255).astype(np.uint8)

        # Compute quality metrics
        original_f = frames.astype(np.float64) / 255.0
        perturbed_f = result.astype(np.float64) / 255.0
        mse = np.mean((original_f - perturbed_f) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        metrics = {
            "final_cosine_similarity": round(final_cos_sim, 6),
            "psnr_db": round(float(psnr), 2),
            "mse": round(float(mse), 8),
            "epsilon": round(self.epsilon * 255, 1),
            "steps": self.steps,
            "best_loss": round(best_loss, 6)
        }

        return result, metrics

    def perturb_video(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 4
    ) -> Dict:
        """
        Load video, adversarially perturb all frames, write output.

        Args:
            input_path: Source video file path
            output_path: Destination for perturbed video (video stream only, no audio)
            batch_size: Frames per optimization batch (lower = less RAM)

        Returns:
            Dict of aggregate metrics across all batches
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise RuntimeError("Video has no readable frames")

        frame_array = np.stack(frames, axis=0)  # (T, H, W, C)

        # Process in batches
        all_perturbed = []
        all_metrics = []

        for i in range(0, len(frame_array), batch_size):
            batch = frame_array[i:i + batch_size]
            perturbed_batch, batch_metrics = self.perturb_batch(batch)
            all_perturbed.append(perturbed_batch)
            all_metrics.append(batch_metrics)

        result_array = np.concatenate(all_perturbed, axis=0)

        # Write output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in result_array:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # Aggregate metrics
        avg_psnr = np.mean([m["psnr_db"] for m in all_metrics])
        avg_cos_sim = np.mean([m["final_cosine_similarity"] for m in all_metrics])
        avg_mse = np.mean([m["mse"] for m in all_metrics])

        return {
            "total_frames": total_frames,
            "batches_processed": len(all_metrics),
            "avg_psnr_db": round(float(avg_psnr), 2),
            "avg_cosine_similarity": round(float(avg_cos_sim), 6),
            "avg_mse": round(float(avg_mse), 8),
            "epsilon": round(self.epsilon * 255, 1),
            "steps_per_batch": self.steps,
            "per_batch_metrics": all_metrics
        }
