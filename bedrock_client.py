import boto3
import yaml
import json

class BedrockClient:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.model_id = config["model"]["modelId"]
        self.content_type = config["model"]["contentType"]
        self.accept = config["model"]["accept"]
        self.parameters = config["model"]["parameters"]
        self.region = config["model"]["region"]

        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def generate_response(self, prompt: str) -> str:
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
