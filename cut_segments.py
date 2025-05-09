import re
import os
import subprocess

ANALYSIS_FILE = "1_analysis.txt"
VIDEO_FILE = "test/1.mp4"
OUTPUT_DIR = "output_segments"

SEGMENT_PATTERN = re.compile(r"Segment (\d+): ([0-9]+\.?[0-9]*)s - ([0-9]+\.?[0-9]*)s ")

def parse_segments(analysis_path):
    segments = []
    with open(analysis_path, "r") as f:
        for line in f:
            match = SEGMENT_PATTERN.search(line)
            if match:
                seg_num = int(match.group(1))
                start = float(match.group(2))
                end = float(match.group(3))
                segments.append((seg_num, start, end))
    return segments

def cut_video_segments(video_path, segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    segment_files = []
    for seg_num, start, end in segments:
        duration = end - start
        output_path = os.path.join(output_dir, f"segment_{seg_num:02d}.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start), "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac", "-strict", "-2", output_path
        ]
        print(f"Cutting segment {seg_num}: {start}s - {end}s -> {output_path}")
        subprocess.run(cmd, check=True)
        segment_files.append(output_path)
    return segment_files

def concat_segments(segment_files, output_path, srt_path=None):
    # Create a temporary file list for ffmpeg concat
    list_path = os.path.join(os.path.dirname(output_path), "segments.txt")
    with open(list_path, "w") as f:
        for seg in segment_files:
            f.write(f"file '{os.path.abspath(seg)}'\n")
    # Run ffmpeg concat with re-encoding, and add subtitles if srt_path is provided
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c:v", "libx264", "-c:a", "aac", "-strict", "-2"
    ]
    if srt_path:
        cmd.extend(["-vf", f"subtitles={srt_path}"])
    cmd.append(output_path)
    print(f"Combining segments into {output_path}" + (f" with subtitles {srt_path}" if srt_path else ""))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    segments = parse_segments(ANALYSIS_FILE)
    print(f"Found {len(segments)} segments.")
    segment_files = cut_video_segments(VIDEO_FILE, segments, OUTPUT_DIR)
    srt_path = ANALYSIS_FILE.replace('.txt', '.srt')
    combined_path = os.path.join(OUTPUT_DIR, "combined.mp4")
    if os.path.exists(srt_path):
        concat_segments(segment_files, combined_path, srt_path)
        print(f"Combined video with subtitles saved to {combined_path}")
    else:
        concat_segments(segment_files, combined_path)
        print(f"Combined video saved to {combined_path}")
