#!/usr/bin/env python3
"""
Badminton Video Analysis Agent

This script uses the OpenAI Agents SDK with Azure OpenAI to analyze badminton videos
and identify segments where players are actively playing.
"""

import json
import os
import sys
import time
import base64
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import tempfile
from io import BytesIO
from dataclasses import dataclass

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import typer
from tqdm import tqdm
from openai import AsyncAzureOpenAI

from agents import Agent, Runner, OpenAIChatCompletionsModel, AgentOutputSchema, function_tool, set_default_openai_client, set_tracing_disabled

# Load environment variables from .env file
load_dotenv()

# Configure Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# Configure Azure OpenAI client
client = AsyncAzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

set_tracing_disabled(disabled=True)

# Maximum number of frames to process in a single batch
MAX_BATCH_SIZE = 50

# Define video processing functions
def extract_frames(video_path: str, sample_rate: int = 5) -> List[Tuple[float, np.ndarray]]:
    """
    Extract frames from a video at a given sample rate.
    
    Args:
        video_path: Path to the video file
        sample_rate: Extract 1 frame every N frames
        
    Returns:
        List of tuples containing (timestamp, frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_count = 0
    
    print(f"Video stats: {total_frames} total frames, {fps:.2f} FPS")
    print(f"Using sample rate: 1 frame every {sample_rate} frames")
    expected_samples = total_frames // sample_rate
    print(f"Expected to extract approximately {expected_samples} frames")
    
    with tqdm(desc="Extracting frames", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Get timestamp in seconds
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
                pbar.update(sample_rate)
                
            frame_count += 1
    
    cap.release()
    print(f"Actually extracted {len(frames)} frames")
    return frames

def encode_frame(frame: np.ndarray) -> str:
    """
    Encode a frame as a base64 string.
    
    Args:
        frame: The frame to encode
        
    Returns:
        Base64 encoded string
    """
    # Convert from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame to reduce size (optional)
    height, width = rgb_frame.shape[:2]
    if width > 800:
        scale_factor = 800 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
    
    # Convert to PIL Image and then to base64
    pil_img = Image.fromarray(rgb_frame)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_frame_by_timestamp(timestamp: float) -> dict:
    """
    Extract a single frame from the video at the given timestamp (in seconds).
    Uses the global VIDEO_PATH set at runtime.
    Args:
        timestamp: Time in seconds to extract the frame
    Returns:
        A dict in the format of {"type": "input_image", "image_url": ..., "detail": "auto"}
    """
    global VIDEO_PATH
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame at {timestamp} seconds (frame {frame_number})")
    # Encode as base64 image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    from PIL import Image
    from io import BytesIO
    import base64
    pil_img = Image.fromarray(rgb_frame)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=80)
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}

@function_tool
def get_frame_by_timestamp_tool(timestamp: float) -> dict:
    """
    Tool: Extract a single frame from the video at the given timestamp (in seconds).
    Returns a dict in the format of {"type": "input_image", "image_url": ..., "detail": "auto"}
    """
    return get_frame_by_timestamp(timestamp)

# Define initial content for badminton rally detection
init_content = [
    {
        "type": "input_text",
        "text": "This is a double badminton game, four players are playing. Both forehand and backhand serves are possible. Frames are taken every second. Only look at the closer court."
    },
    {
        "type": "input_text",
        "text": (
            "For each segment, only include frames where the shuttlecock is in play and all players are actively rallying. "
            "Exclude frames where the rally is clearly paused, players are retrieving the shuttle, or waiting. "
            "If the shuttle is not visible for more than 6 frames, consider the rally paused. "
            "A segment MUST start with serving or preparing to receive."
        )
    },
]

# Define the data model for rally information
@dataclass
class RallyInfo:
    start_frame: int
    start_time: float
    end_frame: int
    end_time: float
    description: str  # Short description of the rally

@dataclass
class BadmintonOutput:
    rallies: List[RallyInfo]
    """A list of rallies with start/end frame numbers and times and descriptions."""

    reason: str
    """Reason for the analysis, e.g., "Rallies detected in the video."""


# Define the Badminton Analysis Agent
class BadmintonAnalysisAgent:
    def __init__(self, model_name: str = "gpt-4.1"):
        """
        Initialize the badminton analysis agent.
        
        Args:
            model_name: Azure OpenAI model to use
        """
        self.agent = Agent(
            name="Badminton Rally Detector",
            instructions="""
            You are an expert badminton video analyzer. Your task is to identify rallies in a series of frame images.

            For each rally, generate a short, vivid, and engaging commentary in the style of a sports commentator, describing what happens in the rally IN CHINESE (e.g., '双方开局谨慎，连续多拍后由后场劈杀得分').
            Output this commentary in the 'description' field for each rally. Do not explain your reasoning or how you determined the rally boundaries—focus only on describing the on-court action as if you are narrating for an audience.

            You MUST strictly follow these rules for segmenting rallies:
            - A segment MUST start with serving or preparing to receive. This rule applies to both your output and the user input.
            - Frames are in a rally if the shuttlecock is in play (it may be out of frame if too high).
            - A segment MUST end when the shuttlecock is falling to the ground.
            - If the shuttle is falling to ground, someone is picking up the shuttle, someone is getting a new shuttle off the court, or the shuttle is out of sight for more than 5 frames, it is NOT a rally.
            - If you are not sure whether frames belong to the same segment, split them. If there is any short or suspected pause, split into multiple segments. If uncertain, err on the side of splitting into more segments.
            """,
            tools=[get_frame_by_timestamp_tool],
            model=OpenAIChatCompletionsModel(openai_client=client, model=model_name),
            output_type=AgentOutputSchema(BadmintonOutput, strict_json_schema=True)
        )
    
    async def analyze_frame_batch(self, frames: List[Tuple[int, float, np.ndarray]]) -> Dict[str, Any]:
        """
        Analyze a batch of video frames to detect rallies.
        
        Args:
            frames: A batch of (frame_number, timestamp, frame) tuples to analyze
            
        Returns:
            Dictionary with 'rallies' information for the batch
        """
        batch_content = init_content.copy()
        
        # Add frames to the content
        for frame_number, timestamp, frame in frames:
            # Encode the frame
            frame_base64 = encode_frame(frame)
            
            batch_content.append({"type": "input_text", "text": f"Frame {frame_number}, Timestamp {timestamp}:"})
            batch_content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{frame_base64}", "detail": "auto"})
        
        # Run the agent with the batch of frames
        result = await Runner.run(
            self.agent,
            [{"role": "user", "content": batch_content}]
        )
        print(result.final_output.reason)
        
        # Convert dataclass to dictionary format for compatibility with existing code
        rallies = []
        for rally in result.final_output.rallies:
            rallies.append({
                "start_frame": rally.start_frame,
                "start_time": rally.start_time,
                "end_frame": rally.end_frame,
                "end_time": rally.end_time,
                "description": rally.description
            })
            
        return {"rallies": rallies}

    async def analyze_video(self, video_path: str, sample_rate: int = 30) -> List[Dict[str, Any]]:
        """
        Analyze a badminton video to identify rallies that span across batches.
        
        Args:
            video_path: Path to the video file
            sample_rate: Sample every N frames
            
        Returns:
            List of rally segments with start and end frame numbers
        """
        # Extract frames from the video
        frame_tuples = extract_frames(video_path, sample_rate)
        
        # Add frame numbers to the frame tuples
        indexed_frames = [(i+1, timestamp, frame) for i, (timestamp, frame) in enumerate(frame_tuples)]
        
        print(f"Processing {len(indexed_frames)} frames in batches of up to {MAX_BATCH_SIZE}")
        
        # Process frames in batches with improved logic to prevent splitting rallies
        all_rallies = []
        i = 0
        last_rally_end = None
        last_call = False
        
        while i < len(indexed_frames) and not last_call:
            # Get the current batch of frames
            batch = indexed_frames[i:i+MAX_BATCH_SIZE]
            print(f"Processing frames {batch[0][0]} to {batch[-1][0]}")
            
            if batch[-1][0] == indexed_frames[-1][0]:
                last_call = True
            
            try:
                # Call the agent to analyze the current batch
                batch_result = await self.analyze_frame_batch(batch)
                batch_rallies = batch_result.get("rallies", [])
                print(batch_rallies)
                
                # Fix: If any rally's 'end_frame' exceeds the batch, set it to batch[-1][0]
                for rally in batch_rallies:
                    if rally["end_frame"] > batch[-1][0]:
                        rally["end_frame"] = batch[-1][0]
                        rally["end_time"] = batch[-1][1]
                        
                if batch_rallies:
                    # If the last rally in this batch ends at the last frame and this is not the last batch,
                    # don't include it yet as it might continue into the next batch
                    if batch_rallies[-1]["end_frame"] == batch[-1][0] and not last_call:
                        all_rallies.extend(batch_rallies[:-1])
                    else:
                        all_rallies.extend(batch_rallies)
                        
                    if all_rallies:
                        # Find the frame index after the last detected rally
                        last_rally_end_idx = next((idx for idx, (frame_num, _, _) in enumerate(indexed_frames) 
                                                if frame_num > all_rallies[-1]["end_frame"]), None)
                        if last_rally_end_idx is not None:
                            if last_rally_end_idx == i:
                                i = batch_rallies[-1]["start_frame"]
                            else:
                                i = last_rally_end_idx
                        else:
                            # If we can't find it, just move to the next batch
                            i += MAX_BATCH_SIZE
                    else:
                        # Move to the next batch if no rallies were found
                        i += MAX_BATCH_SIZE
                else:
                    # Move to the next batch if no rallies were found
                    i += MAX_BATCH_SIZE
                    
            except Exception as e:
                import traceback
                print(f"Error processing batch frames: {e}")
                traceback.print_exc()
                raise
        
        # Convert the rallies to a format with timestamps
        rally_segments = []
        for rally in all_rallies:
            rally_segments.append({
                "start_time": rally["start_time"],
                "end_time": rally["end_time"],
                "start_frame": rally["start_frame"],
                "end_frame": rally["end_frame"],
                "description": rally["description"],
                "frames": [frame for frame in indexed_frames if rally["start_frame"] <= frame[0] <= rally["end_frame"]]
            })

        # Write SRT file for unmerged segments
        srt_path = None
        if rally_segments and hasattr(self, 'srt_output_path') and self.srt_output_path:
            srt_path = self.srt_output_path
        elif rally_segments:
            # fallback: use default naming
            srt_path = 'unmerged_segments.srt'
        if rally_segments and srt_path:
            with open(srt_path, "w") as srt:
                def srt_time(t):
                    h = int(t // 3600)
                    m = int((t % 3600) // 60)
                    s = int(t % 60)
                    ms = int((t - int(t)) * 1000)
                    return f"{h:02}:{m:02}:{s:02},{ms:03}"
                for i, seg in enumerate(rally_segments, 1):
                    srt.write(f"{i}\n{srt_time(seg['start_time'])} --> {srt_time(seg['end_time'])}\n{seg['description']}\n\n")

        # Merge consecutive segments if the last frame of previous is the same or one less than the first frame of next
        if rally_segments:
            merged_segments = [rally_segments[0]]
            for seg in rally_segments[1:]:
                prev = merged_segments[-1]
                if prev["end_frame"] == seg["start_frame"] or prev["end_frame"] + 1 == seg["start_frame"]:
                    # Merge segments
                    prev["end_time"] = seg["end_time"]
                    prev["end_frame"] = seg["end_frame"]
                    prev["frames"].extend(seg["frames"])
                else:
                    merged_segments.append(seg)
            rally_segments = merged_segments

        return rally_segments

# CLI application
app = typer.Typer(help="Badminton video analysis tool using Azure OpenAI")

VIDEO_PATH = None

@app.command()
def analyze(
    video_path: str = typer.Argument(..., help="Path to the badminton video file"),
    output_path: Optional[str] = typer.Option(None, help="Path to save analysis results"),
    sample_rate: int = typer.Option(30, help="Sample every N frames"),
):
    global VIDEO_PATH
    VIDEO_PATH = video_path

    """
    Analyze a badminton video to identify segments with active play.
    """
    # Validate video path
    if not os.path.exists(video_path):
        typer.echo(f"Error: Video file not found: {video_path}")
        raise typer.Exit(code=1)
    
    # Create default output path if not provided
    if output_path is None:
        video_name = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_name)[0]
        output_path = f"{video_name_no_ext}_analysis.txt"
    
    # Run the analysis
    typer.echo(f"Analyzing badminton video: {video_path}")
    typer.echo(f"Using Azure OpenAI deployment: {AZURE_OPENAI_DEPLOYMENT}")
    
    try:
        # Initialize the agent
        agent = BadmintonAnalysisAgent()
        
        # Run the analysis asynchronously
        segments = asyncio.run(agent.analyze_video(video_path, sample_rate))
        
        # Display and save results
        typer.echo(f"\nIdentified {len(segments)} segments of active play:")
        with open(output_path, "w") as f, open(output_path.replace(".txt", ".srt"), "w") as srt:
            f.write(f"Badminton Video Analysis Results\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, segment in enumerate(segments, 1):
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                start_frame = segment["start_frame"]
                end_frame = segment["end_frame"]
                duration = end_time - start_time
                description = segment.get("description", "")
                
                segment_info = f"Segment {i}: {start_time:.2f}s - {end_time:.2f}s (Frames: {start_frame}-{end_frame}, Duration: {duration:.2f}s)"
                typer.echo(segment_info)
                f.write(f"{segment_info}\n")
                
                # Write a few sample frames from this segment
                sample_frames = segment["frames"][:3] + segment["frames"][-3:] if len(segment["frames"]) > 6 else segment["frames"]
                for frame_num, timestamp, _ in sample_frames:
                    f.write(f"  - Frame {frame_num} at {timestamp:.2f}s\n")
                f.write("\n")

                # Write SRT entry
                def srt_time(t):
                    h = int(t // 3600)
                    m = int((t % 3600) // 60)
                    s = int(t % 60)
                    ms = int((t - int(t)) * 1000)
                    return f"{h:02}:{m:02}:{s:02},{ms:03}"
                srt.write(f"{i}\n{srt_time(start_time)} --> {srt_time(end_time)}\n{description}\n\n")
        
        typer.echo(f"\nAnalysis results saved to: {output_path}")
        
    except Exception as e:
        typer.echo(f"Error during analysis: {str(e)}")
        raise typer.Exit(code=1)

# Entry point
if __name__ == "__main__":
    app()