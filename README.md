# Badminton Video Analysis Agent

This project provides a tool for analyzing badminton game videos to automatically detect and segment periods of active play (rallies) using Azure OpenAI.

## Features
- Extracts frames from badminton videos at a configurable sample rate
- Uses Azure OpenAI (GPT-4.1) to analyze frames and identify rally segments
- Outputs a summary of detected rallies with timestamps and frame numbers
- Saves results to a text file

## Requirements
- Python 3.8+
- Azure OpenAI access (API key, endpoint, deployment)
- FFmpeg (for video processing)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup
1. Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_API_VERSION`
   - `AZURE_OPENAI_DEPLOYMENT` (default: gpt-4.1)

2. Place your badminton video files in the project directory.

## Usage
Run the analyzer from the command line:

```bash
python badminton_analyzer.py analyze <video_path> [--output-path <output_file>] [--sample-rate <N>] [--min-segment-length <seconds>]
```

- `<video_path>`: Path to the badminton video file (e.g., `output_segments/combined.mp4`)
- `--output-path`: (Optional) Path to save the analysis results
- `--sample-rate`: (Optional) Sample every N frames (default: 30)
- `--min-segment-length`: (Optional) Minimum segment length in seconds (default: 3.0)

Example:
```bash
python badminton_analyzer.py analyze output_segments/combined.mp4 --sample-rate 30 --min-segment-length 3.0
```

## Output
- The tool prints detected rally segments to the console and saves them to a text file (default: `<video_name>_analysis.txt`).
- Each segment includes start/end times, frame numbers, and sample frame information.

## Project Structure
- `badminton_analyzer.py`: Main analysis script and CLI
- `cut_segments.py`: Extracts and combines video segments based on rally timestamps from an analysis file. Parses segment times from the analysis output, cuts those segments from the source video using ffmpeg, and saves both individual segment files and a combined video.
  
  **Sample command:**
  ```bash
  python cut_segments.py
  ```
- `output_segments/`: Contains generated video segments and the combined video
- `requirements.txt`: Python dependencies
- `test/`: Test files

## Notes
- The tool uses Azure OpenAI GPT-4.1 to analyze batches of video frames. Make sure your API quota is sufficient.
- For best results, use high-quality, stable badminton game footage.

## License
MIT License
