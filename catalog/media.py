"""Reject clearly non-image catalog assets even when retailers label them images."""
from pathlib import PurePosixPath
from urllib.parse import urlparse


def is_non_image_url(url):
    return PurePosixPath(urlparse(url).path).suffix.lower() in {'.mp4', '.webm', '.mov', '.m4v', '.pdf', '.svg'}
