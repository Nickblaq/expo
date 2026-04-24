# yt-dlp FastAPI Server v2

Python + FastAPI server powered by yt-dlp. No Node.js, no binaries to manage.
yt-dlp runs as a Python library directly — no subprocess, no PATH issues.

-----

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

-----

## Endpoints — all browser callable

### `GET /`

Health check.

```
http://localhost:8000/
```

-----

### `GET /info?url=`

Full video metadata + all formats, classified and grouped.

```
http://localhost:8000/info?url=https://www.youtube.com/watch?v=asBeyqk_zpk
```

**Response includes:**

- `id`, `title`, `description`, `uploader`, `uploader_url`, `channel_id`
- `upload_date` (YYYYMMDD), `timestamp` (unix), `duration`, `duration_string`
- `view_count`, `like_count`, `comment_count`, `age_limit`
- `categories`, `tags`, `chapters`
- `is_live`, `was_live`
- `thumbnail`, `thumbnails` (top 5 by resolution)
- `has_subtitles`, `has_auto_captions`, `subtitle_languages`, `auto_caption_languages`
- `format_count`, `formats` (flat list), `formats_grouped` (combined / video_only / audio_only)

**Each format includes:**
`format_id`, `ext`, `type`, `resolution`, `width`, `height`, `fps`,
`vcodec`, `acodec`, `tbr`, `vbr`, `abr`, `asr`, `filesize`, `filesize_human`,
`format_note`, `protocol`, `dynamic_range`

-----

### `GET /formats?url=`

Formats only — skips the full metadata. Faster if you already have metadata.

```
http://localhost:8000/formats?url=https://www.youtube.com/watch?v=asBeyqk_zpk
```

-----

### `GET /transcript?url=&lang=`

Returns the transcript as structured JSON segments with `start`, `duration`, `text`.
Tries manual subtitles first, falls back to auto-captions automatically.

```
http://localhost:8000/transcript?url=https://www.youtube.com/watch?v=asBeyqk_zpk
http://localhost:8000/transcript?url=...&lang=fr
```

**Response:**

```json
{
  "success": true,
  "data": {
    "video_id": "asBeyqk_zpk",
    "title": "...",
    "language": "en",
    "type": "auto",
    "segment_count": 142,
    "full_text": "Hello and welcome to...",
    "segments": [
      { "start": 0.0, "duration": 3.2, "text": "Hello and welcome" },
      { "start": 3.2, "duration": 2.1, "text": "to this video" }
    ]
  }
}
```

-----

### `POST /download`

Downloads a video or audio file to the server. Returns the filename.

**Body options:**

```json
{ "url": "...", "quality": "720p", "ext": "mp4" }
{ "url": "...", "quality": "1080p", "ext": "mp4" }
{ "url": "...", "quality": "audio", "ext": "mp3" }
{ "url": "...", "format_id": "137" }
```

|Field    |Options                                  |
|---------|-----------------------------------------|
|url      |YouTube URL (required)                   |
|format_id|Specific format_id from /formats         |
|quality  |best · 1080p · 720p · 480p · 360p · audio|
|ext      |mp4 · webm · m4a · mp3                   |

**Response:**

```json
{
  "success": true,
  "data": {
    "title": "Video Title",
    "filename": "Video Title [a1b2].mp4",
    "filesize_human": "45.3MB",
    "fetch_url": "/download/file?path=Video+Title+%5Ba1b2%5D.mp4"
  }
}
```

-----

### `GET /download/file?path=`

Serves a downloaded file by filename. Use the `fetch_url` from the download response.

```
http://localhost:8000/download/file?path=Video+Title+%5Ba1b2%5D.mp4
```

-----

## Deploy on Railway

`requirements.txt` is all Railway needs — it detects Python automatically.

Set start command in Railway to:

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Railway provides ffmpeg via nixpacks if you add:

```toml
# nixpacks.toml
[phases.setup]
nixPkgs = ['ffmpeg']
```

ffmpeg is needed for merging video+audio streams (1080p+) and MP3 extraction.

-----

## Notes

- Downloads are saved to `./downloads/` (configurable via `DOWNLOADS_DIR` env var)
- Transcript requires the video to have subtitles or auto-captions enabled
- yt-dlp runs in a thread pool — server stays responsive during long downloads
- All yt-dlp exceptions are caught and returned as structured JSON errors
