"""
Video Analysis API Endpoint
Integrates with existing GoatVision backend
"""

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import tempfile
import shutil
from typing import Optional
import sys

sys.path.insert(0, 'D:/weight_detect')
from goat_video_analyzer import GoatVideoAnalyzer

router = APIRouter(prefix="/api/video", tags=["video"])

# Initialize analyzer once
analyzer = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = GoatVideoAnalyzer()
    return analyzer

@router.post("/analyze")
async def analyze_video_endpoint(
    video: UploadFile = File(...),
    assumed_distance_m: Optional[float] = Form(1.5),
    skip_frames: Optional[int] = Form(3)
):
    """
    Analyze uploaded video for goat detection, weight estimation, and behavior.

    Parameters:
    - video: Video file (MP4, AVI, etc.)
    - assumed_distance_m: Assumed distance to goat in meters (default: 1.5)
    - skip_frames: Process every Nth frame for speed (default: 3)

    Returns:
    - JSON with analysis results and path to annotated video
    """

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        shutil.copyfileobj(video.file, tmp_video)
        video_path = tmp_video.name

    # Output path for annotated video
    output_path = video_path.replace('.mp4', '_analyzed.mp4')

    try:
        # Analyze video
        analyzer_instance = get_analyzer()
        results = analyzer_instance.analyze_video(
            video_path,
            output_path=output_path,
            assumed_distance_m=assumed_distance_m,
            skip_frames=skip_frames
        )

        if results['status'] == 'success':
            # Move annotated video to media folder
            media_dir = Path('D:/weight_detect/output/web_media')
            media_dir.mkdir(exist_ok=True, parents=True)

            final_video_path = media_dir / f"analyzed_{Path(video.filename).stem}.mp4"
            shutil.move(output_path, str(final_video_path))

            results['output_video'] = f"/media/{final_video_path.name}"

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': str(e)
            }
        )
    finally:
        # Cleanup temp file
        if Path(video_path).exists():
            Path(video_path).unlink()

@router.get("/status")
def video_analysis_status():
    """Check if video analysis system is ready"""
    try:
        analyzer_instance = get_analyzer()
        return {
            "status": "ready",
            "detection_model": "yolo11n",
            "weight_model": "pooled",
            "features": [
                "goat_detection",
                "weight_estimation",
                "angle_classification",
                "behavior_detection"
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
