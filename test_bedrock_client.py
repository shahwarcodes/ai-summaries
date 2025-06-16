"""
Unit tests for the BedrockClient class.
Tests the functionality of text generation using mocked AWS Bedrock responses.
"""

import unittest
from unittest.mock import patch, MagicMock
from bedrock_client import BedrockClient

class TestBedrockClient(unittest.TestCase):
    """Test cases for BedrockClient functionality."""
    
    @patch("boto3.client")
    def test_generate_response(self, mock_boto_client):
        """
        Test the generate_response method with a mocked Bedrock response.
        Verifies that the client correctly processes and returns the completion.
        """
        # Mock Bedrock response
        mock_client = MagicMock()
        mock_response = {
            "body": MagicMock(read=MagicMock(return_value=b'{"completion": "This is a test."}'))
        }
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client

        # Initialize and test
        client = BedrockClient("config.yaml")
        result = client.generate_response("Test prompt")

        self.assertEqual(result, "This is a test.")

if __name__ == "__main__":
    unittest.main()
