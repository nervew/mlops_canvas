import logging
import os
import tempfile
from typing import Optional

from azure.storage.blob import BlobServiceClient


logger = logging.getLogger(__name__)


class BlobDownloader:
    """Utility to download assets from Azure Blob Storage."""

    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self._client = BlobServiceClient.from_connection_string(connection_string)
        self._container = self._client.get_container_client(container_name)

    def download_to_temp(self, blob_path: str) -> str:
        """Download a blob to a temporary file and return its path."""

        logger.info("Downloading blob %s from container %s", blob_path, self.container_name)
        blob_client = self._container.get_blob_client(blob_path)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            download_stream = blob_client.download_blob()
            temp_file.write(download_stream.readall())
            temp_file.flush()
            logger.debug("Blob %s stored at %s", blob_path, temp_file.name)
            return temp_file.name

    def download_text(self, blob_path: str, encoding: str = "utf-8") -> str:
        """Download blob content as text."""

        logger.info("Downloading text blob %s from container %s", blob_path, self.container_name)
        blob_client = self._container.get_blob_client(blob_path)
        download_stream = blob_client.download_blob()
        content = download_stream.readall().decode(encoding)
        return content


__all__ = ["BlobDownloader"]
