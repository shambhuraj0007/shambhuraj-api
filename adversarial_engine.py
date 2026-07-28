"""
Adversarial Perturbation Engine
===============================
Gradient-optimized frame perturbation module for testing the robustness
of perceptual hashing and deep feature detection algorithms.

Includes:
  - Classical perceptual hashing (DCT pHash, Block dHash)
  - Deep Neural Feature Extractor (Differentiable CNN spatial feature maps)
  - Short-Time Fourier Transform (STFT) audio landmark spectral extraction
"""

import numpy as np
import torch
import torch.nn as functional_F
import torch.nn.functional as F
import cv2
from typing import Callable, Optional, Dict, Tuple


class DifferentiableHashExtractor:
    """
    Differentiable approximations of perceptual hash functions and spatial feature maps.
    Enables gradient-based adversarial optimization against classical and neural feature matchers.
    """

    @staticmethod
    def dct_phash(frame: torch.Tensor, hash_size: int = 16) -> torch.Tensor:
        """
        Differentiable DCT-based perceptual hash (pHash).

        Args:
            frame: (B, C, H, W) tensor normalized to [0, 1]
            hash_size: Output hash dimension (hash_size x hash_size bits)
        """
        B, C, H, W = frame.shape

        # Luma conversion (Rec. 601)
        gray = (0.299 * frame[:, 0:1, :, :]
                + 0.587 * frame[:, 1:2, :, :]
                + 0.114 * frame[:, 2:3, :, :])

        dct_size = hash_size * 2
        resized = F.interpolate(
            gray, size=(dct_size, dct_size),
            mode='bilinear', align_corners=False
        )

        dct_mat = DifferentiableHashExtractor._build_dct_matrix(dct_size, frame.device)
        dct_result = dct_mat @ resized.squeeze(1) @ dct_mat.T

        low_freq = dct_result[:, :hash_size, :hash_size]
        flat = low_freq.reshape(B, -1)
        median = flat.median(dim=1, keepdim=True)[0]
        hash_bits = torch.sigmoid(10.0 * (flat - median))

        return hash_bits

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
        """
        B, C, H, W = frame.shape
        gray = (0.299 * frame[:, 0:1, :, :]
                + 0.587 * frame[:, 1:2, :, :]
                + 0.114 * frame[:, 2:3, :, :])

        pooled = F.adaptive_avg_pool2d(gray, (hash_size, hash_size))
        diffs = pooled[:, :, :, 1:] - pooled[:, :, :, :-1]
        hash_bits = torch.sigmoid(10.0 * diffs)

        return hash_bits.reshape(B, -1)

    @staticmethod
    def deep_neural_embeddings(frame: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Deep Spatial Feature Map (Multi-scale Spatial Pyramids).
        Simulates deep visual feature representations (e.g. CNN / ViT embedding projections).
        """
        B, C, H, W = frame.shape
        # Multi-scale spatial pooling representations
        p1 = F.adaptive_avg_pool2d(frame, (4, 4)).reshape(B, -1)
        p2 = F.adaptive_avg_pool2d(frame, (8, 8)).reshape(B, -1)
        p3 = F.adaptive_max_pool2d(frame, (4, 4)).reshape(B, -1)
        
        embeddings = torch.cat([p1, p2, p3], dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    @staticmethod
    def combined_hash(frame: torch.Tensor) -> torch.Tensor:
        """
        Comprehensive feature vector combining classical perceptual hashes and deep spatial embeddings.
        """
        phash = DifferentiableHashExtractor.dct_phash(frame, hash_size=16)
        bhash = DifferentiableHashExtractor.block_hash(frame, hash_size=12)
        neural = DifferentiableHashExtractor.deep_neural_embeddings(frame)
        return torch.cat([phash, bhash, neural], dim=1)


class STFTAudioFingerprinter:
    """
    Short-Time Fourier Transform (STFT) audio landmark spectral extractor.
    Enables frequency domain landmark benchmarking.
    """

    @staticmethod
    def compute_spectrogram(audio_signal: torch.Tensor, n_fft: int = 512, hop_length: int = 256) -> torch.Tensor:
        """
        Computes STFT magnitude spectrogram for audio tensors.
        """
        window = torch.hann_window(n_fft, device=audio_signal.device)
        stft = torch.stft(
            audio_signal,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        magnitude = torch.abs(stft)
        return magnitude


class AdversarialPerturber:
    """
    Applies gradient-optimized perturbations to video frames to maximize feature divergence
    within an imperceptible epsilon bound.
    """

    def __init__(
        self,
        target_hash_fn: Optional[Callable] = None,
        epsilon: float = 8.0,
        steps: int = 40,
        learning_rate: float = 0.01
    ):
        self.hash_fn = target_hash_fn or DifferentiableHashExtractor.combined_hash
        self.epsilon = epsilon / 255.0
        self.steps = steps
        self.lr = learning_rate
        self.device = torch.device("cpu")

    def perturb_batch(
        self,
        frames: np.ndarray,
        original_hashes: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, Dict]:
        B, H, W, C = frames.shape

        clean = torch.from_numpy(frames).float().to(self.device) / 255.0
        clean = clean.permute(0, 3, 1, 2)

        if original_hashes is None:
            with torch.no_grad():
                original_hashes = self.hash_fn(clean).detach()

        delta = torch.zeros_like(clean, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        best_loss = float('inf')
        best_delta = delta.data.clone()

        for step in range(self.steps):
            optimizer.zero_grad()

            perturbed = (clean + delta).clamp(0.0, 1.0)
            current_hash = self.hash_fn(perturbed)

            cos_sim = F.cosine_similarity(current_hash, original_hashes, dim=1)
            loss = cos_sim.mean()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data.clamp_(-self.epsilon, self.epsilon)
                delta.data = (clean + delta.data).clamp(0.0, 1.0) - clean

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.data.clone()

        with torch.no_grad():
            final = (clean + best_delta).clamp(0.0, 1.0)
            final_hash = self.hash_fn(final)
            final_cos_sim = F.cosine_similarity(final_hash, original_hashes, dim=1).mean().item()

        result = final.permute(0, 2, 3, 1).cpu().numpy()
        result = (result * 255.0).clip(0, 255).astype(np.uint8)

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
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise RuntimeError("Video has no readable frames")

        frame_array = np.stack(frames, axis=0)

        all_perturbed = []
        all_metrics = []

        for i in range(0, len(frame_array), batch_size):
            batch = frame_array[i:i + batch_size]
            perturbed_batch, batch_metrics = self.perturb_batch(batch)
            all_perturbed.append(perturbed_batch)
            all_metrics.append(batch_metrics)

        result_array = np.concatenate(all_perturbed, axis=0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in result_array:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

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
