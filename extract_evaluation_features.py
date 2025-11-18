"""Extract emotion2vec features for evaluation dataset (fully following emilia_vevo_integration)"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features(audio_path, model, target_sr=16000):
    """Extract features from single audio"""
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # funasr requires filepath
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
            torchaudio.save(temp_path, waveform, target_sr)
        
        try:
            result = model.generate(temp_path, granularity="frame")
            return result[0]['feats']
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        logger.error(f"Failed {audio_path}: {e}")
        return None

def process_ravdess(ravdess_dir, output_dir, model):
    ravdess_dir = Path(ravdess_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(ravdess_dir.glob("*.wav"))
    logger.info(f"RAVDESS: {len(wav_files)} files")
    
    for wav in tqdm(wav_files, desc="RAVDESS"):
        feat_file = output_dir / wav.name.replace('.wav', '_ev2_frame.npy')
        if feat_file.exists():
            continue
        
        features = extract_features(wav, model)
        if features is not None:
            np.save(feat_file, features)
            
            # RAVDESS labels
            parts = wav.stem.split('-')
            if len(parts) >= 3:
                emo_map = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 
                          5:'angry', 6:'fearful', 7:'disgust', 8:'surprised'}
                label_file = output_dir / wav.name.replace('.wav', '_emotion.txt')
                with open(label_file, 'w') as f:
                    f.write(emo_map.get(int(parts[2]), 'unknown'))

def process_esd(esd_dir, output_dir, model):
    esd_dir = Path(esd_dir) / "Emotion Speech Dataset"
    output_dir = Path(output_dir)
    
    wav_files = list(esd_dir.glob("*/*.wav"))
    logger.info(f"ESD: {len(wav_files)} files")
    
    for wav in tqdm(wav_files, desc="ESD"):
        rel_path = wav.relative_to(esd_dir)
        feat_file = output_dir / rel_path.parent / (wav.stem + '_ev2_frame.npy')
        feat_file.parent.mkdir(parents=True, exist_ok=True)
        
        if feat_file.exists():
            continue
        
        features = extract_features(wav, model)
        if features is not None:
            np.save(feat_file, features)
            
            label_file = feat_file.parent / (wav.stem + '_emotion.txt')
            with open(label_file, 'w') as f:
                f.write(wav.parent.name.lower())

def main():
    logger.info("="*60)
    logger.info("Feature extraction")
    logger.info("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load model (fully following emilia_vevo_integration)
    from funasr import AutoModel
    logger.info("Loading emotion2vec...")
    model = AutoModel(model="emotion2vec/emotion2vec_base", hub="hf", device=str(device))
    logger.info("✅ Model loaded successfully")
    
    # Use environment variables or relative paths (user configures via local_config.sh)
    data_root = Path(os.environ.get("DATASET_ROOT", "./raw_data"))
    output_root = Path(os.environ.get("EVAL_FEATURES_ROOT", "./data"))
    
    logger.info("\nRAVDESS")
    process_ravdess(data_root / "RAVDESS", output_root / "RAVDESS", model)
    
    logger.info("\nESD")
    process_esd(data_root / "ESD", output_root / "ESD", model)
    
    logger.info("\n✅ Complete")

if __name__ == "__main__":
    main()

