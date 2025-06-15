from flask import Flask, request, jsonify
from bedrock_client import BedrockClient

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
        prompt = f"""\n\nHuman: Summarize this customer issue in one line:\n"{message}"\n\nAssistant:"""
        summary = bedrock.generate_response(prompt)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
