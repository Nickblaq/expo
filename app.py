"""
app.py — Advanced yt-dlp FastAPI server

Endpoints:
  GET  /                          — health check
  GET  /info?url=                 — full video metadata + all formats
  GET  /formats?url=              — formats only (lightweight)
  GET  /transcript?url=&lang=     — transcript/subtitles as structured JSON
  POST /download                  — download video/audio, returns the file
  GET  /download/file?path=       — serve a previously downloaded file

Design:
  - All yt-dlp work runs in a ThreadPoolExecutor (yt-dlp is sync/blocking)
    so FastAPI's async event loop is never blocked
  - Downloads are saved to ./downloads/, served back via FileResponse
  - Transcripts are extracted in-memory (no file written to disk)
  - Format classification mirrors what you'd expect: combined, video-only, audio-only
"""

import os
import re
import asyncio
import tempfile
import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import yt_dlp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Setup ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="yt-dlp server", version="2.0.0")

DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "./downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# yt-dlp is CPU/IO-bound and synchronous — run it in a thread pool
executor = ThreadPoolExecutor(max_workers=4)


# ── Models ─────────────────────────────────────────────────────────────────────

class DownloadRequest(BaseModel):
    url: str
    format_id: Optional[str] = None        # specific format_id from /formats
    quality: Optional[str] = "best"        # best | 1080p | 720p | 480p | 360p | audio
    ext: Optional[str] = "mp4"             # mp4 | webm | m4a | mp3


# ── Helpers ────────────────────────────────────────────────────────────────────

BASE_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
    "noplaylist": True,
    "socket_timeout": 15,
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    },
}


def classify_format(f: dict) -> dict | None:
    """
    Converts a raw yt-dlp format dict into a clean, classified object.
    Returns None for storyboard/thumbnail formats that aren't playable.
    """
    vcodec = f.get("vcodec", "none")
    acodec = f.get("acodec", "none")
    has_video = vcodec not in (None, "none")
    has_audio = acodec not in (None, "none")

    if not has_video and not has_audio:
        return None

    if has_video and has_audio:
        kind = "video+audio"
    elif has_video:
        kind = "video-only"
    else:
        kind = "audio-only"

    filesize = f.get("filesize") or f.get("filesize_approx")

    return {
        "format_id": f.get("format_id"),
        "ext": f.get("ext"),
        "type": kind,
        "resolution": f.get("resolution") or (
            f"{f['height']}p" if f.get("height") else "audio only"
        ),
        "width": f.get("width"),
        "height": f.get("height"),
        "fps": f.get("fps"),
        "vcodec": vcodec if has_video else None,
        "acodec": acodec if has_audio else None,
        "tbr": f.get("tbr"),                    # total bitrate kbps
        "vbr": f.get("vbr"),                    # video bitrate kbps
        "abr": f.get("abr"),                    # audio bitrate kbps
        "asr": f.get("asr"),                    # audio sample rate hz
        "filesize": filesize,
        "filesize_human": _human_bytes(filesize),
        "format_note": f.get("format_note"),
        "protocol": f.get("protocol"),
        "dynamic_range": f.get("dynamic_range"),  # SDR, HDR10, HLG etc.
    }


def _human_bytes(b: int | None) -> str:
    if not b:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"


def _extract_video_id(url: str) -> str | None:
    """Extracts YouTube video ID from any known URL format."""
    patterns = [
        r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _resolve_format_selector(
    format_id: str | None,
    quality: str | None,
    ext: str | None,
) -> str:
    """Turns user-facing options into a yt-dlp format selector string."""
    if format_id:
        return f"{format_id}+bestaudio/{format_id}"

    quality_map = {
        "best":  "bestvideo+bestaudio/best",
        "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "720p":  "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "480p":  "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "360p":  "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "audio": "bestaudio/best",
    }

    selector = quality_map.get(quality or "best", "bestvideo+bestaudio/best")

    # Narrow by ext if specified and not audio-only
    if ext and ext != "mp3" and quality != "audio":
        height = quality.replace("p", "") if quality and "p" in quality else None
        if height:
            selector = (
                f"bestvideo[height<={height}][ext={ext}]+bestaudio"
                f"/bestvideo[height<={height}]+bestaudio/best"
            )

    return selector


# ── Sync worker functions (run in thread pool) ─────────────────────────────────

def _fetch_info(url: str) -> dict:
    opts = {
        **BASE_OPTS,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        raw = ydl.extract_info(url, download=False)
        raw = ydl.sanitize_info(raw)

    formats = [classify_format(f) for f in raw.get("formats", [])]
    formats = [f for f in formats if f is not None]

    thumbnails = sorted(
        raw.get("thumbnails") or [],
        key=lambda t: (t.get("width") or 0) * (t.get("height") or 0),
        reverse=True,
    )

    return {
        "id": raw.get("id"),
        "title": raw.get("title"),
        "description": (raw.get("description") or "")[:500] or None,
        "uploader": raw.get("uploader"),
        "uploader_url": raw.get("uploader_url"),
        "channel_id": raw.get("channel_id"),
        "upload_date": raw.get("upload_date"),           # YYYYMMDD string
        "timestamp": raw.get("timestamp"),               # unix epoch
        "duration": raw.get("duration"),                 # seconds
        "duration_string": raw.get("duration_string"),
        "view_count": raw.get("view_count"),
        "like_count": raw.get("like_count"),
        "comment_count": raw.get("comment_count"),
        "age_limit": raw.get("age_limit"),
        "categories": raw.get("categories"),
        "tags": (raw.get("tags") or [])[:20],            # cap at 20 tags
        "is_live": raw.get("is_live"),
        "was_live": raw.get("was_live"),
        "chapters": raw.get("chapters"),
        "thumbnail": raw.get("thumbnail"),
        "thumbnails": thumbnails[:5],                    # top 5 by resolution
        "webpage_url": raw.get("webpage_url"),
        "playability_status": raw.get("availability"),
        "has_subtitles": bool(raw.get("subtitles")),
        "has_auto_captions": bool(raw.get("automatic_captions")),
        "subtitle_languages": list((raw.get("subtitles") or {}).keys()),
        "auto_caption_languages": list(
            (raw.get("automatic_captions") or {}).keys()
        )[:10],
        "format_count": len(formats),
        "formats": formats,
        "formats_grouped": {
            "combined": [f for f in formats if f["type"] == "video+audio"],
            "video_only": [f for f in formats if f["type"] == "video-only"],
            "audio_only": [f for f in formats if f["type"] == "audio-only"],
        },
    }


def _fetch_transcript(url: str, lang: str) -> dict:
    """
    Extracts subtitle/transcript data in-memory using yt-dlp.
    Returns structured segments with start, duration, text.
    Falls back to auto-captions if manual subtitles are unavailable.
    """
    opts = {
        **BASE_OPTS,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3",     # json3 gives structured start/duration/text
    }

    transcript_data = {}

    with yt_dlp.YoutubeDL(opts) as ydl:
        raw = ydl.extract_info(url, download=False)
        raw = ydl.sanitize_info(raw)

        # Check manual subtitles first, then fall back to auto-captions
        subtitles = raw.get("subtitles") or {}
        auto_captions = raw.get("automatic_captions") or {}

        source = None
        source_type = None

        if lang in subtitles:
            source = subtitles[lang]
            source_type = "manual"
        elif lang in auto_captions:
            source = auto_captions[lang]
            source_type = "auto"
        else:
            # Try finding any English variant (en-US, en-GB etc.)
            for key in list(subtitles.keys()) + list(auto_captions.keys()):
                if key.startswith(lang):
                    source = subtitles.get(key) or auto_captions.get(key)
                    source_type = "manual" if key in subtitles else "auto"
                    lang = key
                    break

        if not source:
            available = list(subtitles.keys()) + list(auto_captions.keys())
            raise ValueError(
                f"No transcript for lang '{lang}'. "
                f"Available: {available or 'none'}"
            )

        # Find the json3 format entry (has structured data)
        json3_entry = next(
            (s for s in source if s.get("ext") == "json3"), None
        )

        if not json3_entry or not json3_entry.get("url"):
            raise ValueError("Transcript URL not found in yt-dlp response")

        # Fetch the actual json3 subtitle file
        with urllib.request.urlopen(json3_entry["url"], timeout=10) as resp:
            raw_json = json.loads(resp.read().decode("utf-8"))

        # Parse json3 format: events -> segs -> utf8
        segments = []
        for event in raw_json.get("events", []):
            if "segs" not in event:
                continue
            start_ms = event.get("tStartMs", 0)
            duration_ms = event.get("dDurationMs", 0)
            text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
            if text and text != "\n":
                segments.append({
                    "start": round(start_ms / 1000, 2),
                    "duration": round(duration_ms / 1000, 2),
                    "text": text,
                })

        # Also build a plain text version
        full_text = " ".join(s["text"] for s in segments)

        transcript_data = {
            "video_id": raw.get("id"),
            "title": raw.get("title"),
            "language": lang,
            "type": source_type,
            "segment_count": len(segments),
            "full_text": full_text,
            "segments": segments,
        }

    return transcript_data


def _run_download(url: str, format_id: str | None, quality: str, ext: str) -> dict:
    """
    Downloads the video/audio to DOWNLOADS_DIR.
    Returns metadata including the final file path.
    """
    selector = _resolve_format_selector(format_id, quality, ext)

    # Unique ID in filename to avoid collisions
    uid = os.urandom(4).hex()
    output_template = str(DOWNLOADS_DIR / f"%(title)s [{uid}].%(ext)s")

    final_path = {"value": None}

    def progress_hook(d: dict):
        if d["status"] == "finished":
            final_path["value"] = d.get("filename") or d.get("info_dict", {}).get("filepath")

    opts = {
        **BASE_OPTS,
        "format": selector,
        "outtmpl": output_template,
        "progress_hooks": [progress_hook],
        "merge_output_format": ext if ext in ("mp4", "mkv", "webm") else "mp4",
        # Extract audio if mp3 requested
        **(
            {
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }]
            }
            if ext == "mp3" or quality == "audio"
            else {}
        ),
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info = ydl.sanitize_info(info)

    # If progress hook didn't catch the filename, find it by uid
    if not final_path["value"]:
        matches = list(DOWNLOADS_DIR.glob(f"*{uid}*"))
        matches = [f for f in matches if not f.suffix in (".part", ".ytdl", ".tmp")]
        if matches:
            final_path["value"] = str(matches[0])

    if not final_path["value"] or not Path(final_path["value"]).exists():
        raise FileNotFoundError("Downloaded file not found on disk")

    file_path = Path(final_path["value"])
    return {
        "title": info.get("title"),
        "file_path": str(file_path),
        "filename": file_path.name,
        "ext": file_path.suffix.lstrip("."),
        "filesize": file_path.stat().st_size,
        "filesize_human": _human_bytes(file_path.stat().st_size),
        "format_selector": selector,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "yt-dlp-server", "version": "2.0.0"}


@app.get("/info")
async def video_info(url: str):
    """
    Returns full video metadata + all classified formats.

    Browser call:
      GET /info?url=https://www.youtube.com/watch?v=VIDEO_ID
    """
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, _fetch_info, url)
        return {"success": True, "data": data}
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/formats")
async def video_formats(url: str):
    """
    Lightweight — returns only the formats list without full metadata.
    Useful when you already have metadata and just need formats.

    Browser call:
      GET /formats?url=https://www.youtube.com/watch?v=VIDEO_ID
    """
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, _fetch_info, url)
        return {
            "success": True,
            "data": {
                "id": data["id"],
                "title": data["title"],
                "format_count": data["format_count"],
                "formats": data["formats"],
                "formats_grouped": data["formats_grouped"],
            },
        }
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/transcript")
async def video_transcript(url: str, lang: str = "en"):
    """
    Returns the transcript/subtitles as structured JSON.
    Tries manual subtitles first, falls back to auto-captions.

    Browser call:
      GET /transcript?url=https://www.youtube.com/watch?v=VIDEO_ID
      GET /transcript?url=...&lang=fr
    """
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            executor, _fetch_transcript, url, lang
        )
        return {"success": True, "data": data}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/download")
async def download_video(req: DownloadRequest):
    """
    Downloads a video or audio file to the server's downloads directory.
    Returns the filename and a path you can use to fetch the file.

    Body:
      { "url": "...", "quality": "720p", "ext": "mp4" }
      { "url": "...", "format_id": "137" }
      { "url": "...", "quality": "audio", "ext": "mp3" }

    After download, fetch the file via:
      GET /download/file?path=<filename>
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            _run_download,
            req.url,
            req.format_id,
            req.quality or "best",
            req.ext or "mp4",
        )
        return {
            "success": True,
            "data": {
                **result,
                "fetch_url": f"/download/file?path={result['filename']}",
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/download/file")
async def serve_file(path: str):
    """
    Serves a previously downloaded file by filename.

    Browser call:
      GET /download/file?path=Some+Title+[abc1].mp4
    """
    file_path = DOWNLOADS_DIR / path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {path}")

    # Security: ensure it's inside downloads dir
    if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
        raise HTTPException(403, "Access denied")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )
