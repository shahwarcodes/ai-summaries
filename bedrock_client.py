"""
AWS Bedrock client wrapper for interacting with foundation models.
Provides a simplified interface for generating text completions using AWS Bedrock.
"""

import boto3
import yaml
import json

class BedrockClient:
    """
    Client for interacting with AWS Bedrock foundation models.
    Handles configuration loading and provides methods for text generation.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the Bedrock client with configuration from a YAML file.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load model configuration
        self.model_id = config["model"]["modelId"]
        self.content_type = config["model"]["contentType"]
        self.accept = config["model"]["accept"]
        self.parameters = config["model"]["parameters"]
        self.region = config["model"]["region"]

        # Initialize AWS Bedrock client
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def generate_response(self, prompt: str) -> str:
        """
        Generate a text completion using the configured Bedrock model.
        
        Args:
            prompt (str): The input prompt for text generation
            
        Returns:
            str: The generated text completion
        """
        request_body = {
            "prompt": prompt,
            **self.parameters
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType=self.content_type,
            accept=self.accept,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        return response_body["completion"].strip()
