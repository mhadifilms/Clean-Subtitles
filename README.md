# Clean Subtitles

## What is this?
This is a tool that transcribes audio from any media file and generates clean SRT subtitles. It uses the highest quality Whisper model available and provides customizable settings for subtitle formatting.

## Features
- **Universal format support**: Works with any audio/video format that FFmpeg supports
- **High-quality transcription**: Uses Whisper large-v2 model for best accuracy
- **Smart line splitting**: Never splits words, respects character limits
- **Customizable settings**: Edit settings.json to customize your preferences
- **Batch processing**: No interactive prompts - perfect for automation

## Settings
The script uses a `settings.json` file for configuration. Edit this file to customize:

```json
{
  "model": "large-v2",
  "lower_case": true,
  "simplify_punctuation": true,
  "single_lines": true,
  "max_chars": 25,
  "min_duration": 1.2,
  "gap_frames": 0
}
```

### Settings Explained
- **model**: Whisper model to use (options: tiny, base, small, medium, large, large-v2)
- **lower_case**: Convert text to lowercase
- **simplify_punctuation**: Remove extra punctuation marks (dialogue mode)
- **single_lines**: Use single line per subtitle (false = double lines)
- **max_chars**: Maximum characters per subtitle line (7-72)
- **min_duration**: Minimum subtitle duration in seconds (1.2-6.0)
- **gap_frames**: Gap between captions in frames (0-10)

## How to use
To get started, run:
```bash
brew install ffmpeg
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python script.py path/to/your/file.mp4
```

The script will automatically create a `settings.json` file with default settings on first run. Edit this file to customize your preferences.

## Examples
```bash
# Simple usage
python script.py video.mp4

# Paths with spaces (no quotes needed!)
python script.py /Volumes/external/My Videos/movie.mkv

# Complex paths work too
python script.py "My Videos" "Summer Vacation" movie.mp4
```

## Output
The script generates an SRT file in the same directory as your input file, with the same name but `.srt` extension.
