from flask import Flask, request, jsonify
from bedrock_client import BedrockClient
from opensearch_rag import retrieve_context

app = Flask(__name__)
bedrock = BedrockClient("config.yaml")

@app.route('/')
def home():
    return jsonify({"message": "Bedrock summarization API is running."})

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        message = data.get("message")

        if not message:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        # Claude uses Anthropic-style prompts
        context = retrieve_context(message)
        context_text = "\n".join(f"- {c}" for c in context)

        prompt = f"""\n\nHuman: The following are past support tickets from this customer:
                {context_text}
                New message:
                "{message}"
                Summarize the customer's issue in one line.\n\nAssistant:"""

        summary = bedrock.generate_response(prompt)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Flask running on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
