"""
Flask application that provides a REST API for text summarization with context-aware responses.
Uses AWS Bedrock for summarization and OpenSearch for retrieving relevant context.
"""

from flask import Flask, request, jsonify
from bedrock_client import BedrockClient
from opensearch_rag import retrieve_context

# Initialize Flask application and Bedrock client
app = Flask(__name__)
bedrock = BedrockClient("config.yaml")

@app.route('/')
def home():
    """Health check endpoint that returns API status."""
    return jsonify({"message": "Bedrock summarization API is running."})

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Endpoint for context-aware text summarization.
    
    Flow:
    1. Receives customer message
    2. Retrieves relevant context from past tickets
    3. Generates a summary using Claude via Bedrock
    
    Returns:
        JSON response containing the summary or error message
    """
    try:
        # Extract message from request
        data = request.get_json()
        message = data.get("message")

        if not message:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        # Get relevant context from past tickets
        context = retrieve_context(message)
        context_text = "\n".join(f"- {c}" for c in context)

        # Construct prompt with context and new message
        prompt = f"""\n\nHuman: The following are past support tickets from this customer:
                {context_text}
                New message:
                "{message}"
                Summarize the customer's issue in one line.\n\nAssistant:"""

        # Generate summary using Claude
        summary = bedrock.generate_response(prompt)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Flask running on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
