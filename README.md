# Text Summarization API

A Flask-based REST API that uses AWS Bedrock to generate concise summaries of customer messages. This service is particularly useful for quickly understanding the essence of customer communications.

## Features

- RESTful API endpoint for text summarization
- Integration with AWS Bedrock's Claude model
- Simple and efficient text processing
- Error handling and input validation

## Prerequisites

- Python 3.7+
- AWS Account with Bedrock access
- AWS credentials configured

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:
   - Place your AWS credentials in `bedrock-user_accessKeys.csv`
   - Ensure the credentials have access to AWS Bedrock

5. Configure the model settings in `config.yaml`:
```yaml
model:
  modelId: "anthropic.claude-v2"
  contentType: "application/json"
  accept: "application/json"
  region: "us-east-1"
  parameters:
    max_tokens_to_sample: 100
    temperature: 0.7
    top_k: 250
    top_p: 1
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

3. Send a POST request to `/summarize` endpoint:
```bash
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{"message": "Your text to summarize here"}'
```

Example response:
```json
{
  "summary": "Customer reports being overcharged 40 times for an invoice and expresses frustration."
}
```

## API Endpoints

### GET /
Health check endpoint that returns the API status.

### POST /summarize
Summarizes the provided text.

Request body:
```json
{
  "message": "Text to summarize"
}
```

Response:
```json
{
  "summary": "Generated summary"
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful summarization
- 400: Missing or invalid input
- 500: Server error

## Testing

Run the test suite:
```bash
python -m unittest test_bedrock_client.py
```

## Security Notes

- Never commit AWS credentials to version control
- Keep your `config.yaml` secure
- Use environment variables for sensitive configuration in production

## License

[TBD]
