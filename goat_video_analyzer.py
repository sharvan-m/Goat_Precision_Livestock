"""
Complete Video Analysis System for Goats
- Detection and tracking
- Weight estimation from video
- Behavior classification
- Best angle detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
from ultralytics import YOLO
from collections import defaultdict
import json

class GoatVideoAnalyzer:
    """Complete goat video analysis system"""

    def __init__(self,
                 detection_model_path: str = "yolo11n.pt",
                 weight_model_path: str = "goat_metric_yw/final_area_pooled.joblib"):

        # Load detection model
        print("Loading YOLO detection model...")
        self.detector = YOLO(detection_model_path)

        # Load weight model
        print("Loading weight estimation model...")
        import joblib
        self.weight_model_data = joblib.load(weight_model_path)
        self.weight_coef = self.weight_model_data['coef']
        self.focal_length = self.weight_model_data.get('fx', 1920)

        print("✓ Models loaded successfully")

        # Tracking data
        self.tracks = defaultdict(list)
        self.track_id_counter = 0

    def predict_weight(self, area_cm2: float) -> float:
        """Predict weight from area in cm²"""
        if area_cm2 <= 0:
            return 0
        return np.exp(self.weight_coef[0] + self.weight_coef[1] * np.log(area_cm2))

    def calculate_area_cm2(self, pixel_area: int, distance_mm: float) -> float:
        """Convert pixel area to cm² using distance"""
        scale = distance_mm / self.focal_length / 10  # mm to cm
        return pixel_area * (scale ** 2)

    def estimate_distance_from_height(self, pixel_height: int, typical_goat_height_cm: float = 55) -> float:
        """Estimate distance from goat's pixel height (rough approximation)"""
        # distance = (real_height × focal_length) / pixel_height
        # For baby goats, use smaller typical height (30cm)
        # Use adaptive height based on pixel size
        if pixel_height < 200:
            typical_goat_height_cm = 30  # Baby goat
        elif pixel_height < 400:
            typical_goat_height_cm = 45  # Young goat
        else:
            typical_goat_height_cm = 55  # Adult goat

        distance_mm = (typical_goat_height_cm * self.focal_length * 10) / pixel_height

        # Clamp to reasonable range (0.5m to 5m)
        distance_mm = max(500, min(5000, distance_mm))
        return distance_mm

    def detect_goats(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect goats in frame"""
        results = self.detector(frame, conf=0.25, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                # Calculate metrics
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 0

                # Create mask from bbox
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'width': width,
                    'height': height,
                    'pixel_area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'mask': mask
                })

        return detections

    def classify_angle(self, detection: Dict) -> Tuple[str, float]:
        """
        Classify viewing angle quality for weight estimation.
        Returns: (angle_type, quality_score)

        Side view is best for weight estimation.
        """
        aspect_ratio = detection['aspect_ratio']

        # Side view: goat appears wider (aspect_ratio > 1.2)
        # Front/back view: goat appears taller (aspect_ratio < 0.8)
        # Oblique: in between

        if aspect_ratio > 1.3:
            return "side_view", 0.95
        elif aspect_ratio > 1.0:
            return "oblique_good", 0.75
        elif aspect_ratio > 0.8:
            return "oblique_fair", 0.50
        else:
            return "front_back_view", 0.20

    def classify_behavior(self, detections_history: List[List[Dict]], window: int = 30) -> str:
        """
        Classify goat behavior based on recent detections.

        Behaviors:
        - standing: minimal movement, vertical orientation
        - eating: head lowered, minimal horizontal movement
        - walking: horizontal movement
        - lying: very low height/width ratio
        """
        if len(detections_history) < window:
            return "unknown"

        recent = detections_history[-window:]

        if not recent or not recent[0]:
            return "unknown"

        # Calculate movement
        centers = [d[0]['center'] if d else (0, 0) for d in recent]
        movements = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            movements.append(np.sqrt(dx**2 + dy**2))

        avg_movement = np.mean(movements) if movements else 0

        # Calculate average height and aspect ratio
        heights = [d[0]['height'] if d else 0 for d in recent]
        aspects = [d[0]['aspect_ratio'] if d else 0 for d in recent]

        avg_height = np.mean(heights)
        avg_aspect = np.mean(aspects)

        # Classification rules
        if avg_aspect > 2.0 or avg_height < 100:
            return "lying"
        elif avg_movement < 5:
            if avg_aspect < 1.2:
                return "standing"
            else:
                return "eating"
        elif avg_movement > 15:
            return "walking"
        else:
            return "standing"

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict],
                       weights: List[float], angles: List[Tuple[str, float]],
                       behavior: str = None) -> np.ndarray:
        """Draw annotations on frame"""
        annotated = frame.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            angle_type, angle_score = angles[i]
            weight = weights[i]

            # Color based on angle quality
            if angle_score > 0.8:
                color = (0, 255, 0)  # Green - good angle
            elif angle_score > 0.5:
                color = (0, 255, 255)  # Yellow - fair angle
            else:
                color = (0, 0, 255)  # Red - poor angle

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw info
            label = f"Weight: {weight:.1f}kg"
            angle_label = f"Angle: {angle_type.replace('_', ' ').title()}"
            conf_label = f"Conf: {det['confidence']:.2f}"

            cv2.putText(annotated, label, (x1, y1 - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(annotated, angle_label, (x1, y1 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(annotated, conf_label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw behavior
        if behavior:
            behavior_text = f"Behavior: {behavior.replace('_', ' ').title()}"
            cv2.putText(annotated, behavior_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return annotated

    def analyze_video(self, video_path: str,
                     output_path: Optional[str] = None,
                     assumed_distance_m: float = 1.5,
                     skip_frames: int = 2) -> Dict[str, Any]:
        """
        Complete video analysis pipeline.

        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video
            assumed_distance_m: Assumed distance to goat (meters)
            skip_frames: Process every Nth frame (speeds up processing)

        Returns:
            Analysis results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Analyzing video: {Path(video_path).name}")
        print(f"{'='*60}\n")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {'error': 'Could not open video'}

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
        print(f"Processing every {skip_frames} frames...")

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps/skip_frames, (width, height))

        # Analysis data
        frame_results = []
        all_detections = []
        all_weights = []
        all_angles = []
        detection_history = []

        frame_idx = 0
        processed_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for speed
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # Detect goats
            detections = self.detect_goats(frame)

            if detections:
                # Estimate weights
                weights = []
                angles = []

                for det in detections:
                    # Classify angle
                    angle_type, angle_score = self.classify_angle(det)
                    angles.append((angle_type, angle_score))

                    # Estimate distance (rough)
                    distance_mm = self.estimate_distance_from_height(
                        det['height'],
                        typical_goat_height_cm=55
                    )

                    # Or use assumed distance
                    if assumed_distance_m:
                        distance_mm = assumed_distance_m * 1000

                    # Calculate weight
                    area_cm2 = self.calculate_area_cm2(det['pixel_area'], distance_mm)
                    weight = self.predict_weight(area_cm2)
                    weights.append(weight)

                    all_weights.append(weight)
                    all_angles.append((angle_type, angle_score))

                # Classify behavior
                detection_history.append(detections)
                behavior = self.classify_behavior(detection_history)

                # Annotate frame
                annotated = self.annotate_frame(frame, detections, weights, angles, behavior)

                # Save frame result
                frame_results.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'num_goats': len(detections),
                    'weights': weights,
                    'angles': [a[0] for a in angles],
                    'angle_scores': [a[1] for a in angles],
                    'behavior': behavior
                })
            else:
                annotated = frame
                detection_history.append([])

            # Write frame
            if writer:
                writer.write(annotated)

            processed_count += 1
            if processed_count % 30 == 0:
                print(f"Processed {processed_count} frames...")

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        # Calculate summary statistics
        if all_weights:
            avg_weight = np.mean(all_weights)
            median_weight = np.median(all_weights)
            std_weight = np.std(all_weights)

            # Count angles
            angle_counts = {}
            for angle_type, _ in all_angles:
                angle_counts[angle_type] = angle_counts.get(angle_type, 0) + 1

            best_angle = max(angle_counts.items(), key=lambda x: x[1])[0]

            # Get weights from best angle frames only
            best_angle_weights = [
                w for w, (a, _) in zip(all_weights, all_angles)
                if a in ['side_view', 'oblique_good']
            ]

            if best_angle_weights:
                recommended_weight = np.median(best_angle_weights)
                weight_range = (
                    np.percentile(best_angle_weights, 20),
                    np.percentile(best_angle_weights, 80)
                )
            else:
                recommended_weight = median_weight
                weight_range = (
                    np.percentile(all_weights, 20),
                    np.percentile(all_weights, 80)
                )

            summary = {
                'status': 'success',
                'video_info': {
                    'path': video_path,
                    'frames': total_frames,
                    'processed_frames': processed_count,
                    'fps': fps,
                    'duration_sec': total_frames / fps
                },
                'detection_summary': {
                    'frames_with_goats': len([r for r in frame_results if r['num_goats'] > 0]),
                    'total_detections': len(all_weights),
                    'avg_goats_per_frame': np.mean([r['num_goats'] for r in frame_results])
                },
                'weight_analysis': {
                    'recommended_weight_kg': round(recommended_weight, 1),
                    'weight_range_kg': (round(weight_range[0], 1), round(weight_range[1], 1)),
                    'average_weight_kg': round(avg_weight, 1),
                    'median_weight_kg': round(median_weight, 1),
                    'std_deviation_kg': round(std_weight, 1),
                    'confidence': 'good' if std_weight < 10 else 'fair' if std_weight < 20 else 'low'
                },
                'angle_analysis': {
                    'angle_distribution': angle_counts,
                    'best_angle_detected': best_angle,
                    'good_angle_percentage': round(
                        100 * sum(1 for a, s in all_angles if s > 0.7) / len(all_angles), 1
                    ) if all_angles else 0
                },
                'behavior_analysis': {
                    'behaviors_detected': list(set(r['behavior'] for r in frame_results if 'behavior' in r))
                },
                'frame_details': frame_results,
                'output_video': output_path if output_path else None
            }
        else:
            summary = {
                'status': 'no_detections',
                'message': 'No goats detected in video',
                'video_info': {
                    'path': video_path,
                    'frames': total_frames,
                    'fps': fps
                }
            }

        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}\n")

        if summary['status'] == 'success':
            print(f"✓ Recommended Weight: {summary['weight_analysis']['recommended_weight_kg']} kg")
            print(f"✓ Weight Range: {summary['weight_analysis']['weight_range_kg'][0]}-{summary['weight_analysis']['weight_range_kg'][1]} kg")
            print(f"✓ Confidence: {summary['weight_analysis']['confidence']}")
            print(f"✓ Best Angle: {summary['angle_analysis']['best_angle_detected']}")
            print(f"✓ Good Angles: {summary['angle_analysis']['good_angle_percentage']}%")

            if output_path:
                print(f"\n✓ Annotated video saved: {output_path}")

        return summary


def test_on_sample_videos():
    """Test the analyzer on sample videos"""

    # Initialize analyzer
    analyzer = GoatVideoAnalyzer()

    # Find videos
    video_dir = Path("D:/weight_detect/videos")
    videos = list(video_dir.glob("*_2min.mp4"))

    if not videos:
        videos = list(video_dir.glob("*.mp4"))[:3]  # Take first 3 if no 2min clips

    print(f"\nFound {len(videos)} videos to analyze\n")

    results = {}

    for video in videos[:3]:  # Test on first 3
        output_path = str(video).replace('.mp4', '_analyzed.mp4')

        result = analyzer.analyze_video(
            str(video),
            output_path=output_path,
            assumed_distance_m=1.5,  # Assume 1.5m distance
            skip_frames=3  # Process every 3rd frame for speed
        )

        results[video.name] = result

        # Save JSON report
        json_path = str(video).replace('.mp4', '_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Analysis saved: {json_path}\n")

    return results


if __name__ == "__main__":
    print("="*60)
    print("GOAT VIDEO ANALYZER")
    print("="*60)

    results = test_on_sample_videos()

    print("\n" + "="*60)
    print("ALL VIDEOS ANALYZED")
    print("="*60)
