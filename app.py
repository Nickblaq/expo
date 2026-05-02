import os
import re
import asyncio
import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="yt-dlp server", version="2.1.0")

DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "./downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=4)

# ── Models ─────────────────────────────────────────────

class DownloadRequest(BaseModel):
    url: str
    format_id: Optional[str] = None
    quality: Optional[str] = "best"
    ext: Optional[str] = "mp4"


class QuickDownloadRequest(BaseModel):
    url: str


# ── Helpers ────────────────────────────────────────────

BASE_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "socket_timeout": 30,
    "retries": 3,
    "fragment_retries": 3,
    "concurrent_fragment_downloads": 3,
}


def _human_bytes(b: int | None) -> str:
    if not b:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"


def classify_format(f: dict) -> dict | None:
    vcodec = f.get("vcodec", "none")
    acodec = f.get("acodec", "none")

    has_video = vcodec not in (None, "none")
    has_audio = acodec not in (None, "none")

    if not has_video and not has_audio:
        return None

    kind = "video+audio" if has_video and has_audio else (
        "video-only" if has_video else "audio-only"
    )

    filesize = f.get("filesize") or f.get("filesize_approx")

    return {
        "format_id": f.get("format_id"),
        "ext": f.get("ext"),
        "type": kind,
        "resolution": f.get("resolution") or (
            f"{f.get('height')}p" if f.get("height") else "audio only"
        ),
        "filesize": filesize,
        "filesize_human": _human_bytes(filesize),
    }


def _resolve_format_selector(format_id, quality, ext):
    if format_id:
        return f"{format_id}+bestaudio/{format_id}"

    q = (quality or "best").lower()
    e = (ext or "mp4").lower()

    if q == "audio":
        return "bestaudio/best"

    if q in ["1080p", "720p", "480p", "360p"]:
        h = int(q.replace("p", ""))
        return (
            f"bestvideo[height<={h}]+bestaudio/"
            f"best[height<={h}]/"
            f"bestvideo+bestaudio/"
            f"best"
        )

    return "bestvideo+bestaudio/best"


# ── Workers ────────────────────────────────────────────

def _fetch_info(url: str) -> dict:
    opts = {**BASE_OPTS, "skip_download": True}

    with yt_dlp.YoutubeDL(opts) as ydl:
        raw = ydl.extract_info(url, download=False)
        raw = ydl.sanitize_info(raw)

    formats = [classify_format(f) for f in raw.get("formats", [])]
    formats = [f for f in formats if f]

    return {
        "id": raw.get("id"),
        "title": raw.get("title"),
        "duration": raw.get("duration"),
        "uploader": raw.get("uploader"),
        "view_count": raw.get("view_count"),
        "formats": formats,
    }


def _run_download(url, format_id, quality, ext):
    selector = _resolve_format_selector(format_id, quality, ext)

    uid = os.urandom(4).hex()
    output_template = str(DOWNLOADS_DIR / f"%(title)s [{uid}].%(ext)s")

    opts = {
        **BASE_OPTS,
        "format": selector,
        "outtmpl": output_template,
        "merge_output_format": ext if ext in ("mp4", "webm", "mkv") else "mp4",
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    matches = list(DOWNLOADS_DIR.glob(f"*{uid}*"))
    matches = [f for f in matches if f.suffix not in (".part", ".tmp")]

    if not matches:
        raise FileNotFoundError("Download failed")

    file = matches[0]

    return {
        "filename": file.name,
        "ext": file.suffix.lstrip("."),
        "filesize": file.stat().st_size,
        "filesize_human": _human_bytes(file.stat().st_size),
    }


def _run_quick_download(url: str) -> dict:
    uid = os.urandom(4).hex()

    opts = {
        **BASE_OPTS,
        "format": "b",
        "outtmpl": str(DOWNLOADS_DIR / f"%(title)s [{uid}].%(ext)s"),
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:python"],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    matches = [
        f for f in DOWNLOADS_DIR.glob(f"*{uid}*")
        if f.suffix not in (".part", ".tmp")
    ]

    if not matches:
        raise FileNotFoundError("Downloaded file not found")

    file = matches[0]

    return {
        "filename": file.name,
        "ext": file.suffix.lstrip("."),
        "filesize_human": _human_bytes(file.stat().st_size),
    }


# ── Routes ─────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/info")
async def info(url: str):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, _fetch_info, url)
    return {"success": True, "data": data}


@app.post("/download")
async def download(req: DownloadRequest):
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        executor,
        _run_download,
        req.url,
        req.format_id,
        req.quality,
        req.ext,
    )

    return {
        "success": True,
        "data": {
            **result,
            "fetch_url": f"/download/file?path={result['filename']}",
        },
    }


@app.post("/quick")
async def quick(req: QuickDownloadRequest):
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        executor,
        _run_quick_download,
        req.url,
    )

    return {
        "success": True,
        "data": {
            **result,
            "fetch_url": f"/download/file?path={result['filename']}",
        },
    }


@app.get("/download/file")
async def serve(path: str):
    file_path = DOWNLOADS_DIR / path

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(str(file_path), filename=file_path.name)
