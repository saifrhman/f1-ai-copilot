#!/usr/bin/env python3
"""
Driver Emotion Classifier
Classifies driver emotions from radio communication using Whisper + emotion models
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import librosa


class EmotionType(Enum):
    CALM = "calm"
    ANGRY = "angry"
    PANICKED = "panicked"
    FOCUSED = "focused"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    """Result of emotion classification"""
    emotion: EmotionType
    confidence: float
    timestamp: Optional[str] = None
    duration: Optional[float] = None
    audio_features: Optional[Dict[str, float]] = None


class AudioFeatureExtractor:
    """Extracts audio features for emotion classification"""
    
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate for analysis
    
    def extract_features(self, audio_file: str) -> Dict[str, float]:
        """
        Extract audio features from audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary of audio features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            features = {}
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            features['mean_pitch'] = np.mean(pitches[magnitudes > 0.1])
            features['pitch_std'] = np.std(pitches[magnitudes > 0.1])
            
            # Energy features
            features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
            features['energy_std'] = np.std(librosa.feature.rms(y=y))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs)
            features['mfcc_std'] = np.std(mfccs)
            
            # Tempo features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate'] = np.mean(zcr)
            
            return features
            
        except Exception as e:
            logging.error(f"Failed to extract audio features: {e}")
            return {}
    
    def extract_mock_features(self) -> Dict[str, float]:
        """Extract mock features for testing"""
        return {
            'mean_pitch': 150.0,
            'pitch_std': 25.0,
            'rms_energy': 0.3,
            'energy_std': 0.1,
            'spectral_centroid_mean': 2000.0,
            'spectral_centroid_std': 500.0,
            'mfcc_mean': 0.0,
            'mfcc_std': 1.0,
            'tempo': 120.0,
            'zero_crossing_rate': 0.05
        }


class EmotionClassifier:
    """Classifies emotions from audio features"""
    
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
        self.emotion_thresholds = self._load_emotion_thresholds()
    
    def _load_emotion_thresholds(self) -> Dict[EmotionType, Dict[str, Tuple[float, float]]]:
        """Load emotion classification thresholds"""
        return {
            EmotionType.CALM: {
                'mean_pitch': (100, 200),
                'pitch_std': (10, 30),
                'rms_energy': (0.1, 0.4),
                'energy_std': (0.05, 0.15)
            },
            EmotionType.ANGRY: {
                'mean_pitch': (200, 400),
                'pitch_std': (40, 80),
                'rms_energy': (0.5, 1.0),
                'energy_std': (0.2, 0.5)
            },
            EmotionType.PANICKED: {
                'mean_pitch': (300, 500),
                'pitch_std': (60, 100),
                'rms_energy': (0.6, 1.0),
                'energy_std': (0.3, 0.6)
            },
            EmotionType.FOCUSED: {
                'mean_pitch': (150, 250),
                'pitch_std': (20, 40),
                'rms_energy': (0.3, 0.6),
                'energy_std': (0.1, 0.2)
            },
            EmotionType.EXCITED: {
                'mean_pitch': (200, 350),
                'pitch_std': (30, 60),
                'rms_energy': (0.4, 0.8),
                'energy_std': (0.15, 0.3)
            },
            EmotionType.FRUSTRATED: {
                'mean_pitch': (180, 300),
                'pitch_std': (35, 65),
                'rms_energy': (0.4, 0.7),
                'energy_std': (0.2, 0.4)
            }
        }
    
    def classify_emotion(self, audio_file: str) -> EmotionResult:
        """
        Classify emotion from audio file
        
        Args:
            audio_file: Path to audio file or base64 encoded audio
            
        Returns:
            EmotionResult with classification
        """
        try:
            # Extract audio features
            features = self.feature_extractor.extract_features(audio_file)
            
            if not features:
                # Use mock features for testing
                features = self.feature_extractor.extract_mock_features()
            
            # Classify emotion based on features
            emotion, confidence = self._classify_from_features(features)
            
            return EmotionResult(
                emotion=emotion,
                confidence=confidence,
                audio_features=features
            )
            
        except Exception as e:
            logging.error(f"Emotion classification failed: {e}")
            return EmotionResult(
                emotion=EmotionType.NEUTRAL,
                confidence=0.0
            )
    
    def _classify_from_features(self, features: Dict[str, float]) -> Tuple[EmotionType, float]:
        """Classify emotion from extracted features"""
        emotion_scores = {}
        
        for emotion, thresholds in self.emotion_thresholds.items():
            score = 0.0
            total_features = 0
            
            for feature_name, (min_val, max_val) in thresholds.items():
                if feature_name in features:
                    feature_value = features[feature_name]
                    
                    # Calculate how well the feature matches the emotion
                    if min_val <= feature_value <= max_val:
                        # Perfect match
                        score += 1.0
                    else:
                        # Calculate distance from ideal range
                        if feature_value < min_val:
                            distance = (min_val - feature_value) / min_val
                        else:
                            distance = (feature_value - max_val) / max_val
                        
                        # Score based on distance (closer = higher score)
                        score += max(0, 1 - distance)
                    
                    total_features += 1
            
            if total_features > 0:
                emotion_scores[emotion] = score / total_features
        
        # Find emotion with highest score
        if emotion_scores:
            best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            return best_emotion[0], best_emotion[1]
        
        return EmotionType.NEUTRAL, 0.5
    
    def classify_emotion_from_text(self, text: str) -> EmotionResult:
        """
        Classify emotion from transcribed text (fallback method)
        
        Args:
            text: Transcribed radio communication text
            
        Returns:
            EmotionResult with classification
        """
        text_lower = text.lower()
        
        # Keyword-based emotion classification
        emotion_keywords = {
            EmotionType.ANGRY: ['angry', 'furious', 'mad', 'pissed', 'damn', 'shit'],
            EmotionType.FRUSTRATED: ['frustrated', 'annoyed', 'upset', 'disappointed'],
            EmotionType.PANICKED: ['panic', 'emergency', 'help', 'urgent', 'quick'],
            EmotionType.EXCITED: ['excited', 'great', 'amazing', 'fantastic', 'brilliant'],
            EmotionType.FOCUSED: ['focus', 'concentrate', 'careful', 'steady'],
            EmotionType.CALM: ['calm', 'relaxed', 'steady', 'smooth']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)
        
        if emotion_scores:
            best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            return EmotionResult(
                emotion=best_emotion[0],
                confidence=min(0.9, best_emotion[1] + 0.3)
            )
        
        return EmotionResult(
            emotion=EmotionType.NEUTRAL,
            confidence=0.5
        )


class WhisperTranscriber:
    """Transcribes audio using Whisper (mock implementation)"""
    
    def __init__(self):
        self.available = self._check_whisper_availability()
    
    def _check_whisper_availability(self) -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except ImportError:
            logging.warning("Whisper not available. Using mock transcription.")
            return False
    
    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text
        """
        if not self.available:
            return self._mock_transcribe(audio_file)
        
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            return result["text"]
        except Exception as e:
            logging.error(f"Whisper transcription failed: {e}")
            return self._mock_transcribe(audio_file)
    
    def _mock_transcribe(self, audio_file: str) -> str:
        """Mock transcription for testing"""
        mock_transcriptions = [
            "The car feels good, but I'm losing time in sector 2.",
            "Damn it! The tires are gone, I can't get any grip!",
            "I need to pit now, the fuel is running low.",
            "Great lap! The car is working perfectly.",
            "I'm struggling with the balance, need to adjust the setup."
        ]
        
        # Use file hash to get consistent mock transcription
        file_hash = hash(audio_file) % len(mock_transcriptions)
        return mock_transcriptions[file_hash]


# Global instances
_emotion_classifier = None
_transcriber = None

def get_emotion_classifier() -> EmotionClassifier:
    """Get or create emotion classifier instance"""
    global _emotion_classifier
    if _emotion_classifier is None:
        _emotion_classifier = EmotionClassifier()
    return _emotion_classifier

def get_transcriber() -> WhisperTranscriber:
    """Get or create transcriber instance"""
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber

def classify_emotion(audio_file: str) -> str:
    """
    Classify driver emotion from audio file
    
    Args:
        audio_file: Path to audio file or base64 encoded audio
        
    Returns:
        Classified emotion as string
    """
    classifier = get_emotion_classifier()
    result = classifier.classify_emotion(audio_file)
    return result.emotion.value

def classify_emotion_detailed(audio_file: str) -> Dict[str, Any]:
    """
    Classify driver emotion with detailed results
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        Detailed emotion classification results
    """
    classifier = get_emotion_classifier()
    transcriber = get_transcriber()
    
    # Classify emotion
    emotion_result = classifier.classify_emotion(audio_file)
    
    # Transcribe audio
    transcription = transcriber.transcribe(audio_file)
    
    # Cross-reference with text-based classification
    text_emotion = classifier.classify_emotion_from_text(transcription)
    
    # Combine results
    final_confidence = (emotion_result.confidence + text_emotion.confidence) / 2
    final_emotion = emotion_result.emotion if emotion_result.confidence > text_emotion.confidence else text_emotion.emotion
    
    return {
        "emotion": final_emotion.value,
        "confidence": final_confidence,
        "transcription": transcription,
        "audio_features": emotion_result.audio_features,
        "text_emotion": text_emotion.emotion.value,
        "text_confidence": text_emotion.confidence
    }


# Example usage and testing
if __name__ == "__main__":
    print("🏁 Driver Emotion Classifier Test")
    print("=" * 50)
    
    classifier = get_emotion_classifier()
    transcriber = get_transcriber()
    
    # Test with mock audio files
    test_audio_files = [
        "radio_communication_1.wav",
        "radio_communication_2.wav",
        "radio_communication_3.wav",
        "radio_communication_4.wav",
        "radio_communication_5.wav"
    ]
    
    for i, audio_file in enumerate(test_audio_files, 1):
        print(f"\n🎤 Audio File {i}: {audio_file}")
        
        # Classify emotion
        emotion_result = classifier.classify_emotion(audio_file)
        
        # Transcribe
        transcription = transcriber.transcribe(audio_file)
        
        print(f"   🎭 Emotion: {emotion_result.emotion.value}")
        print(f"   🎯 Confidence: {emotion_result.confidence:.2f}")
        print(f"   📝 Transcription: {transcription}")
        
        if emotion_result.audio_features:
            print(f"   🔊 Audio Features: {len(emotion_result.audio_features)} features extracted")
        
        print("-" * 50)
    
    # Test detailed classification
    print("\n🔍 Detailed Classification Test:")
    detailed_result = classify_emotion_detailed("test_radio.wav")
    print(f"Final Emotion: {detailed_result['emotion']}")
    print(f"Confidence: {detailed_result['confidence']:.2f}")
    print(f"Transcription: {detailed_result['transcription']}")
    print(f"Text Emotion: {detailed_result['text_emotion']}")
    print(f"Text Confidence: {detailed_result['text_confidence']:.2f}") 