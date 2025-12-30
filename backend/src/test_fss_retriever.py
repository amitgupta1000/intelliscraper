import asyncio
import unittest
from unittest.mock import patch, AsyncMock, MagicMock, call, ANY
import os
import sys

# Add the source directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fss_retriever import GeminiFileSearchRetriever

class AsyncTestCase(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

class TestGeminiFileSearchRetriever(AsyncTestCase):

    @patch('fss_retriever.genai.Client')
    def setUp(self, mock_genai_client):
        """Set up a fresh retriever and mock client for each test."""
        # Mock the entire genai client and its async counterpart
        self.mock_async_client = AsyncMock()
        mock_genai_client.return_value.aio = self.mock_async_client
        
        # Patch the logger to prevent log file output during tests
        self.patcher = patch('fss_retriever.logger')
        self.mock_logger = self.patcher.start()

        self.retriever = GeminiFileSearchRetriever()

    def tearDown(self):
        """Stop the patcher after each test."""
        self.patcher.stop()

    def test_initialization(self):
        """Test that the retriever initializes with a unique name and client."""
        self.assertIn("crystal-fss-", self.retriever.display_name)
        self.assertIsNotNone(self.retriever.client)
        self.assertEqual(self.retriever.async_client, self.mock_async_client)
        self.assertEqual(self.retriever.created_file_names, [])

    @patch('fss_retriever.tempfile.NamedTemporaryFile')
    @patch('fss_retriever.os')
    def test_upload_single_file_success_with_polling(self, mock_os, mock_tempfile):
        """Test successful file upload, including the operation polling logic."""
        # --- Mock Setup ---
        # Mock tempfile creation
        mock_tf_obj = MagicMock()
        mock_tf_obj.name = "/tmp/fakefile.txt"
        mock_tempfile.return_value = mock_tf_obj
        mock_os.path.exists.return_value = True

        # Mock the initial operation object returned by the upload call
        initial_op = MagicMock()
        initial_op.name = "operations/12345"
        initial_op.done = False

        # Mock the subsequent polled operation objects
        polling_op = MagicMock()
        polling_op.name = "operations/12345"
        polling_op.done = False

        # Mock the final, completed operation object
        final_op = MagicMock()
        final_op.name = "operations/12345"
        final_op.done = True
        
        # The result of the completed operation is a File object
        mock_file_result = MagicMock()
        mock_file_result.name = "files/abcdef"
        final_op.result.return_value = mock_file_result

        # Configure the async client mocks
        self.mock_async_client.file_search_stores.upload_to_file_search_store.return_value = initial_op
        self.mock_async_client.operations.get.side_effect = [polling_op, final_op]

        # --- Test Execution ---
        result = self.run_async(
            self.retriever._upload_single_file("http://example.com", "some content", "stores/test-store")
        )

        # --- Assertions ---
        # Check that the file was written and cleaned up
        mock_tf_obj.write.assert_called_once_with("some content")
        mock_tf_obj.close.assert_called_once()
        mock_os.unlink.assert_called_once_with("/tmp/fakefile.txt")

        # Check that the upload was initiated
        self.mock_async_client.file_search_stores.upload_to_file_search_store.assert_awaited_once_with(
            file="/tmp/fakefile.txt",
            file_search_store_name="stores/test-store",
            config=ANY
        )

        # Check that polling occurred correctly
        self.mock_async_client.operations.get.assert_has_awaits([
            call("operations/12345"),
            call("operations/12345")
        ])
        
        # Check that the final result is the File object from the completed operation
        self.assertEqual(result, mock_file_result)

    def test_upload_single_file_empty_content(self):
        """Test that files with empty or whitespace-only content are skipped."""
        result = self.run_async(
            self.retriever._upload_single_file("http://example.com", "  \n  ", "stores/test-store")
        )
        self.assertIsNone(result)
        self.mock_async_client.file_search_stores.upload_to_file_search_store.assert_not_awaited()
        self.mock_logger.warning.assert_called_once()

    def test_create_and_upload_contexts_success(self):
        """Test the creation of a store and parallel upload of multiple files."""
        # --- Mock Setup ---
        # Mock store creation
        mock_store = MagicMock()
        mock_store.name = "stores/new-store-123"
        self.mock_async_client.file_search_stores.create.return_value = mock_store

        # Mock the result of _upload_single_file
        mock_file_1 = MagicMock()
        mock_file_1.name = "files/file1"
        mock_file_2 = MagicMock()
        mock_file_2.name = "files/file2"

        # --- Test Execution ---
        with patch.object(self.retriever, '_upload_single_file', new_callable=AsyncMock) as mock_upload:
            mock_upload.side_effect = [mock_file_1, mock_file_2]
            
            contexts = {
                "http://a.com": {"content": "content a"},
                "http://b.com": {"content": "content b"}
            }
            store_name = self.run_async(self.retriever.create_and_upload_contexts(contexts))

            # --- Assertions ---
            # Store was created and its name is returned
            self.mock_async_client.file_search_stores.create.assert_awaited_once()
            self.assertEqual(store_name, "stores/new-store-123")
            self.assertEqual(self.retriever.file_store_name, "stores/new-store-123")

            # Uploads were called in parallel
            mock_upload.assert_has_awaits([
                call("http://a.com", "content a", "stores/new-store-123"),
                call("http://b.com", "content b", "stores/new-store-123")
            ])

            # Uploaded file names were tracked for cleanup
            self.assertIn("files/file1", self.retriever.created_file_names)
            self.assertIn("files/file2", self.retriever.created_file_names)
            self.assertEqual(len(self.retriever.created_file_names), 2)

    def test_answer_question_flow(self):
        """Test the end-to-end flow of answering a question."""
        # --- Mock Setup ---
        # Mock the sub-methods that are tested elsewhere
        self.retriever.create_and_upload_contexts = AsyncMock(return_value="stores/test-store")
        self.retriever.delete_store = AsyncMock()

        # Mock the final LLM call
        mock_response = MagicMock()
        mock_response.text = "This is the final answer."
        self.mock_async_client.models.generate_content.return_value = mock_response

        # --- Test Execution ---
        answer = self.run_async(self.retriever.answer_question("test query", {"http://a.com": {"content": "..."}}))

        # --- Assertions ---
        # Check that the answer is correct
        self.assertEqual(answer, "This is the final answer.")

        # Check that the main methods were called
        self.retriever.create_and_upload_contexts.assert_awaited_once()
        self.mock_async_client.models.generate_content.assert_awaited_once()
        
        # Check that cleanup is always called
        self.retriever.delete_store.assert_awaited_once()

    def test_delete_store(self):
        """Test that the store and all tracked files are deleted."""
        # --- Mock Setup ---
        self.retriever.file_store_name = "stores/store-to-delete"
        self.retriever.created_file_names = ["files/file1", "files/file2"]

        # --- Test Execution ---
        self.run_async(self.retriever.delete_store())

        # --- Assertions ---
        # Check that file deletion was attempted for each tracked file
        self.mock_async_client.files.delete.assert_has_awaits([
            call(name="files/file1"),
            call(name="files/file2")
        ], any_order=True)

        # Check that the store itself was deleted
        self.mock_async_client.file_search_stores.delete.assert_awaited_once_with(name="stores/store-to-delete")

        # Check that the internal state is cleared
        self.assertIsNone(self.retriever.file_store_name)
        self.assertEqual(self.retriever.created_file_names, [])

if __name__ == '__main__':
    unittest.main()
