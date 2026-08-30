from abc import ABC, abstractmethod
import numpy as np
import time
from app.utils.logging import logger

class SpeakerEmbeddingBackend(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Loads and warms the model."""
        pass

    @abstractmethod
    def extract_embedding(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extracts the embedding vector from raw samples."""
        pass


class SpeechBrainBackend(SpeakerEmbeddingBackend):
    def __init__(self):
        self.model = None
        self.device = None

    def load_model(self) -> None:
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier

        if self.model is not None:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"SpeechBrainBackend: Loading ECAPA model on {self.device}...")
        
        t0 = time.perf_counter()
        
        # Load pre-trained model weights.
        # LocalStrategy.COPY is required on Windows: SpeechBrain's default SYMLINK
        # strategy needs privileges the service does not have, and its fallback leaves
        # behind text files containing the cache path instead of the real weights.
        from_hparams_kwargs = {
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "savedir": "pretrained_models/spkrec-ecapa-voxceleb",
            "run_opts": {"device": self.device},
        }
        try:
            from speechbrain.utils.fetching import LocalStrategy
            from_hparams_kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass  # older SpeechBrain releases copy by default

        self.model = EncoderClassifier.from_hparams(**from_hparams_kwargs)
        
        load_time = (time.perf_counter() - t0) * 1000.0
        logger.info(f"SpeechBrainBackend: Loaded model in {load_time:.1f}ms. Performing model warming...")
        
        # Model warming: run a dummy 1-second silence inference
        w_t0 = time.perf_counter()
        dummy_tensor = torch.zeros(1, 16000, device=self.device)
        with torch.no_grad():
            _ = self.model.encode_batch(dummy_tensor)
        warm_time = (time.perf_counter() - w_t0) * 1000.0
        
        logger.info(f"SpeechBrainBackend: Model warmed in {warm_time:.1f}ms. Ready.")

    def extract_embedding(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        
        if self.model is None:
            raise RuntimeError("SpeechBrainBackend: Model not loaded.")

        # Ensure samples is a 1D numpy array
        if len(samples.shape) > 1:
            samples = samples.flatten()

        # Convert numpy array to float32 torch tensor on the correct device
        tensor = torch.from_numpy(samples).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # encode_batch returns tensor of shape [batch, time, embedding_dim]
            embedding = self.model.encode_batch(tensor)
            # Squeeze to get [embedding_dim] and convert to numpy array
            return embedding.squeeze(0).squeeze(0).cpu().numpy()
