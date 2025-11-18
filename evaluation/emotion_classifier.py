"""
Emotion classifier - Use emotion2vec for emotion recognition
Copy EmotionClassifierV2 class from emotion_information_bottleneck
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmotionClassifierV2:
    """Emotion classifier V2 - Use emotion2vec 9-class native output"""
    
    def __init__(self, model_name: str, hub: str, device: str):
        self.model_name = model_name
        self.hub = hub
        self.device = device
        self.model = None
        
        # emotion2vec 9-class emotions (correct order, refer to official documentation)
        # 0:angry, 1:disgusted, 2:fearful, 3:happy, 4:neutral, 5:other, 6:sad, 7:surprised, 8:unknown
        self.ev2_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']
        
        # ESD 5-class emotions (as reference ground truth)
        self.esd_emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
        
        # Optional loose mapping (for reference comparison)
        # emotion2vec 9 classes -> ESD 5 classes (only for reference analysis)
        self.ev2_to_esd_loose_mapping = {
            'angry': 'angry',
            'disgusted': 'angry',      # disgust -> angry
            'fearful': 'sad',          # fear -> sad
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
            'surprised': 'surprise',   # Note spelling difference
            'other': None,             # no mapping
            'unknown': None            # no mapping
        }
        
        logger.info(f"Initializing emotion classifier V2 (emotion2vec 9-class native output)")
        logger.info(f"  model: {model_name}")
        logger.info(f"  emotion2vec supports 9 classes: {self.ev2_emotions}")
    
    def load_model(self):
        """Load emotion2vec model"""
        if self.model is not None:
            return
        
        try:
            from funasr import AutoModel
            
            logger.info("Loading emotion2vec full model...")
            
            self.model = AutoModel(
                model=self.model_name,
                hub=self.hub
            )
            
            logger.info("✅ emotion2vec model loaded successfully")
            logger.info("✅ Using 9-class native emotion classification")
            
            # Set standard 9-class emotion labels mapping (eliminate warning)
            emotion_labels = {
                0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 
                4: "neutral", 5: "other", 6: "sad", 7: "surprised", 8: "unknown"
            }
            
            # Try to set to model configuration
            try:
                if hasattr(self.model, 'config'):
                    self.model.config.id2label = emotion_labels
                    self.model.config.label2id = {v: k for k, v in emotion_labels.items()}
                    logger.info("✅ Standard label mapping set to model configuration")
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                    self.model.model.config.id2label = emotion_labels
                    self.model.model.config.label2id = {v: k for k, v in emotion_labels.items()}
                    logger.info("✅ Standard label mapping set to model configuration")
            except Exception as e:
                logger.debug(f"Cannot set model configuration: {e}")
            
            # Verify label order
            try:
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                    retrieved_labels = [self.model.config.id2label[i] for i in range(9)]
                    self.ev2_emotions = retrieved_labels
                    logger.info(f"✅ Label order confirmed: {self.ev2_emotions}")
                elif hasattr(self.model, 'id2label'):
                    self.ev2_emotions = [self.model.id2label[i] for i in range(len(self.model.id2label))]
                    logger.info(f"✅ Label order obtained from model: {self.ev2_emotions}")
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'label_list'):
                    self.ev2_emotions = list(self.model.model.label_list)
                    logger.info(f"✅ Label order obtained from model: {self.ev2_emotions}")
                else:
                    # Use predefined standard order
                    self.ev2_emotions = list(emotion_labels.values())
                    logger.info(f"✅ Using standard label order: {self.ev2_emotions}")
            except Exception as e:
                logger.warning(f"⚠️ Label order verification failed: {e}, using predefined standard order")
                self.ev2_emotions = list(emotion_labels.values())
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    def classify_from_audio(self, audio_path: str, esd_ground_truth: Optional[str] = None) -> Dict:
        """
        Classify from audio file, return emotion2vec 9-class prediction
        
        Args:
            audio_path: audio file path
            esd_ground_truth: ESD dataset ground truth label (optional, for reference)
        
        Returns:
            {
                'ev2_predicted': str,           # emotion2vec predicted emotion (one of 9 classes)
                'ev2_confidence': float,        # prediction confidence
                'ev2_probs': np.ndarray (9,),  # 9-class probability distribution
                'esd_ground_truth': str,       # ESD ground truth label (if provided)
                'esd_mapped': str,             # ESD category obtained through loose mapping
                'is_consistent': bool          # whether mapped result is consistent with ESD ground truth
            }
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Use emotion2vec for classification
            result = self.model.generate(
                audio_path,
                output_dir=None,
                granularity="utterance",
                extract_embedding=False
            )
            
            if result and len(result) > 0:
                # Get emotion2vec raw prediction
                ev2_label_raw = result[0].get('labels', result[0].get('label', 'unknown'))
                # Handle case where labels might be a list
                if isinstance(ev2_label_raw, list):
                    ev2_label_full = ev2_label_raw[0] if len(ev2_label_raw) > 0 else 'unknown'
                else:
                    ev2_label_full = ev2_label_raw
                
                # Handle mixed Chinese-English labels, extract English part (e.g. "生气/angry" -> "angry")
                if '/' in str(ev2_label_full):
                    ev2_label = str(ev2_label_full).split('/')[-1].strip()
                else:
                    ev2_label = str(ev2_label_full)
                
                scores = result[0].get('scores', None)
                
                # If scores is a single value, construct probability distribution
                if scores is not None and not isinstance(scores, (list, np.ndarray)):
                    # Assume this is predicted category confidence
                    ev2_confidence = float(scores)
                    ev2_probs = np.zeros(len(self.ev2_emotions))
                    try:
                        pred_idx = self.ev2_emotions.index(ev2_label)
                        ev2_probs[pred_idx] = ev2_confidence
                        # Distribute remaining probability evenly
                        remaining = (1.0 - ev2_confidence) / (len(self.ev2_emotions) - 1)
                        for i in range(len(self.ev2_emotions)):
                            if i != pred_idx:
                                ev2_probs[i] = remaining
                    except ValueError:
                        # If label not in list, use uniform distribution
                        ev2_probs = np.ones(len(self.ev2_emotions)) / len(self.ev2_emotions)
                        ev2_confidence = 1.0 / len(self.ev2_emotions)
                elif scores is not None:
                    # scores is already probability distribution
                    ev2_probs = np.array(scores)
                    ev2_confidence = ev2_probs.max()
                    # Re-determine label based on probability distribution (fix bug: not using model returned labels)
                    pred_idx = int(ev2_probs.argmax())
                    if pred_idx < len(self.ev2_emotions):
                        ev2_label = self.ev2_emotions[pred_idx]
                else:
                    # No scores, use default high confidence
                    ev2_confidence = 0.9
                    ev2_probs = np.zeros(len(self.ev2_emotions))
                    try:
                        pred_idx = self.ev2_emotions.index(ev2_label)
                        ev2_probs[pred_idx] = ev2_confidence
                        remaining = (1.0 - ev2_confidence) / (len(self.ev2_emotions) - 1)
                        for i in range(len(self.ev2_emotions)):
                            if i != pred_idx:
                                ev2_probs[i] = remaining
                    except ValueError:
                        ev2_probs = np.ones(len(self.ev2_emotions)) / len(self.ev2_emotions)
                        ev2_confidence = 1.0 / len(self.ev2_emotions)
                
                # Convert to ESD category through loose mapping (for reference only)
                esd_mapped = self.ev2_to_esd_loose_mapping.get(ev2_label, None)
                
                # Check consistency
                is_consistent = False
                if esd_ground_truth and esd_mapped:
                    is_consistent = (esd_mapped == esd_ground_truth)
                
                return {
                    'ev2_predicted': ev2_label,
                    'ev2_confidence': float(ev2_confidence),
                    'ev2_probs': ev2_probs,
                    'ev2_all_labels': self.ev2_emotions,
                    'esd_ground_truth': esd_ground_truth,
                    'esd_mapped': esd_mapped,
                    'is_consistent': is_consistent
                }
            else:
                raise ValueError("Classification failed: model returned empty result")
                
        except Exception as e:
            logger.error(f"Classification failed ({audio_path}): {e}")
            raise
    
    def classify_from_features(self, features: np.ndarray, audio_path: Optional[str] = None, 
                              esd_ground_truth: Optional[str] = None) -> Dict:
        """
        Classify from features (scientific experiment, no fallback)
        
        If audio_path is provided, prioritize classification from audio; otherwise must classify from features
        """
        # If audio_path exists, prioritize classification from audio
        if audio_path and os.path.exists(audio_path):
            return self.classify_from_audio(audio_path, esd_ground_truth)
        
        # Classify from features
        if self.model is None:
            self.load_model()
        
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        features = features.to(self.device)
        
        # Average pooling to [1, 1024]
        if features.dim() == 2:  # [T, D]
            pooled_features = features.mean(dim=0, keepdim=False)  # [D]
        else:
            pooled_features = features
        
        if pooled_features.dim() == 1:
            pooled_features = pooled_features.unsqueeze(0)  # [1, D]
        
        # Directly access emotion2vec_plus classification head: model.model.proj
        # According to test results, this is a Linear(1024 -> 9)
        if not (hasattr(self.model, 'model') and hasattr(self.model.model, 'proj')):
            raise RuntimeError("Cannot access emotion2vec_plus classification head model.model.proj")
        
        classifier = self.model.model.proj
        
        # Dimension consistency check
        in_features = int(classifier.in_features)
        if pooled_features.size(-1) != in_features:
            raise RuntimeError(
                f"Feature dimension mismatch: got {pooled_features.size(-1)}, expected {in_features}. "
                "Please ensure using emotion2vec variant that matches classification head or align dimensions at extraction end."
            )
        
        with torch.no_grad():
            # Through classification head
            logits = classifier(pooled_features)  # [1, 9]
            
            if logits.dim() > 1:
                logits = logits.squeeze(0)  # [9]
            
            # Check dimension
            if logits.shape[0] != len(self.ev2_emotions):
                raise RuntimeError(f"Classification head output dimension {logits.shape[0]} and expected {len(self.ev2_emotions)} mismatch")
            
            # Calculate probability
            probs = F.softmax(logits, dim=0)
            
            # Prediction
            pred_idx = probs.argmax().item()
            ev2_confidence = probs[pred_idx].item()
            ev2_predicted = self.ev2_emotions[pred_idx]
            
            # Map to ESD
            esd_mapped = self.ev2_to_esd_loose_mapping.get(ev2_predicted, None)
            is_consistent = False
            if esd_ground_truth and esd_mapped:
                is_consistent = (esd_mapped == esd_ground_truth)
            
            return {
                'ev2_predicted': ev2_predicted,
                'ev2_confidence': ev2_confidence,
                'ev2_probs': probs.cpu().numpy(),
                'ev2_all_labels': self.ev2_emotions,
                'esd_ground_truth': esd_ground_truth,
                'esd_mapped': esd_mapped,
                'is_consistent': is_consistent
            }
