"""
Tests unitaires pour le module d'intégration cloud.
"""

import unittest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from ai_models.cloud_integration import (
    CloudConfig,
    CloudStorage,
    CloudCompute,
    CloudManager,
)


class TestCloudStorage(unittest.TestCase):
    def setUp(self):
        self.config = CloudConfig(
            provider="aws",
            credentials={"access_key": "test_key", "secret_key": "test_secret"},
            region="us-east-1",
            bucket="test-bucket",
            prefix="test",
        )
        self.storage = CloudStorage(self.config)

    @patch("boto3.client")
    async def test_upload_file(self, mock_client):
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile() as tmp:
            path = Path(tmp.name)
            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3

            # Tester l'upload
            url = await self.storage.upload_file(path)

            mock_s3.upload_file.assert_called_once()
            self.assertTrue(url.startswith("s3://"))

    @patch("boto3.client")
    async def test_download_file(self, mock_client):
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3

        with tempfile.NamedTemporaryFile() as tmp:
            path = Path(tmp.name)
            result = await self.storage.download_file("test.obj", path)

            mock_s3.download_file.assert_called_once()
            self.assertEqual(result, path)

    @patch("boto3.client")
    async def test_list_files(self, mock_client):
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        files = await self.storage.list_files()

        mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
        self.assertIsInstance(files, list)


class TestCloudCompute(unittest.TestCase):
    def setUp(self):
        self.config = CloudConfig(
            provider="aws",
            credentials={"access_key": "test_key", "secret_key": "test_secret"},
            region="us-east-1",
            bucket="test-bucket",
            prefix="test",
        )
        self.compute = CloudCompute(self.config)

    @patch("boto3.client")
    async def test_start_instance(self, mock_client):
        mock_ec2 = MagicMock()
        mock_client.return_value = mock_ec2

        mock_ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-12345"}]}

        instance_id = await self.compute.start_instance("t2.micro", "test-instance")

        mock_ec2.run_instances.assert_called_once()
        self.assertEqual(instance_id, "i-12345")

    @patch("boto3.client")
    async def test_stop_instance(self, mock_client):
        mock_ec2 = MagicMock()
        mock_client.return_value = mock_ec2

        await self.compute.stop_instance("i-12345")

        mock_ec2.stop_instances.assert_called_once_with(InstanceIds=["i-12345"])

    @patch("boto3.client")
    async def test_run_job(self, mock_client):
        mock_batch = MagicMock()
        mock_client.return_value = mock_batch

        mock_batch.submit_job.return_value = {"jobId": "job-12345"}

        job_id = await self.compute.run_job(
            {
                "name": "test-job",
                "queue": "test-queue",
                "definition": "test-def",
                "command": ["echo", "test"],
            }
        )

        mock_batch.submit_job.assert_called_once()
        self.assertEqual(job_id, "job-12345")


class TestCloudManager(unittest.TestCase):
    def setUp(self):
        self.config = CloudConfig(
            provider="aws",
            credentials={"access_key": "test_key", "secret_key": "test_secret"},
            region="us-east-1",
            bucket="test-bucket",
            prefix="test",
        )
        self.manager = CloudManager(self.config)

    async def test_sync_project(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)

            # Créer quelques fichiers
            (path / "test1.txt").write_text("test1")
            (path / "test2.txt").write_text("test2")

            with patch.object(self.manager.storage, "list_files") as mock_list:
                mock_list.return_value = []

                with patch.object(self.manager.storage, "upload_file") as mock_upload:
                    await self.manager.sync_project(path)

                    # Vérifier que les fichiers ont été uploadés
                    self.assertEqual(mock_upload.call_count, 2)

    async def test_process_remote(self):
        with patch.object(self.manager.compute, "run_job") as mock_run_job:
            mock_run_job.return_value = "job-12345"

            with patch.object(self.manager.compute, "get_job_status") as mock_status:
                mock_status.return_value = {"status": "SUCCEEDED"}

                with patch.object(
                    self.manager.storage, "download_file"
                ) as mock_download:
                    result = await self.manager.process_remote(
                        "input.obj", "output.obj", {"name": "test-job"}
                    )

                    self.assertEqual(result["job_id"], "job-12345")
                    self.assertEqual(result["status"]["status"], "SUCCEEDED")


def run_tests():
    """Lance les tests unitaires."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
