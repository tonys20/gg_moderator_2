import json
import os
import sys
from mitmproxy import http, ctx

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from ibm_granite.run_granite import GraniteGuardian


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

guardian_model = None
class OpenAIProxy:
    def __init__(self):
        self.api_host = "api.openai.com"
        self.guardian = None

    def load_guardian(self):
        global guardian_model
        if guardian_model is None:
            try:
                ctx.log.info("Loading Granite Guardian model...")
                guardian_model = GraniteGuardian()
                ctx.log.info("Granite Guardian model loaded successfully")
            except Exception as e:
                ctx.log.error(f"Error loading Granite Guardian model: {str(e)}")
        return guardian_model

    def request(self, flow: http.HTTPFlow) -> None:
        if flow.request.headers.get("X-Guardian-Scanned"):
            return
        if flow.request.pretty_host == "localhost":
            flow.request.host = self.api_host
            flow.request.port = 443
            flow.request.scheme = "https"
            ctx.log.info("Rewriting localhost request to api.openai.com")

        if self.api_host in flow.request.pretty_host and "/v1/chat/completions" in flow.request.path:
            try:
                content = flow.request.content.decode('utf-8')
                request_data = json.loads(content)

                messages = request_data.get('messages', [])
                user_messages = [msg["content"] for msg in messages if msg.get("role") == "user"]
                if not user_messages:
                    return

                last_user_message = user_messages[-1]
                ctx.log.info(f"Processing user message: {last_user_message[:100]}...")

                guardian = self.load_guardian()
                if guardian is None:
                    ctx.log.warn("Granite Guardian model not available, allowing request")
                    return

                # Get risk assessment
                flow.request.headers["X-Guardian-Scanned"] = "1"
                results = guardian.classify_all_risks(last_user_message)
                blocked_categories = [ category for category in results.keys()  if results[category]['label'] == 'Yes']

                if blocked_categories:
                    ctx.log.warn(f"Blocked due to: {blocked_categories}")
                    self.block_request(
                        flow,
                        category_name=", ".join(blocked_categories),
                        confidence=max(results[cat]["confidence"] for cat in blocked_categories),
                        matched_terms=blocked_categories
                    )
                    return

            except Exception as e:
                ctx.log.error(f"Error processing request: {str(e)}")

    def response(self, flow: http.HTTPFlow) -> None:

        if self.api_host in flow.request.pretty_host:
            ctx.log.info(f"Intercepted response from {flow.request.pretty_host}")

            try:
                if flow.response and flow.response.content:
                    content = flow.response.content.decode('utf-8')
                    data = json.loads(content)

                    assistant_text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if not assistant_text:
                        return

                    ctx.log.info(f"Analyzing assistant response: {assistant_text[:100]}")

                    guardian = self.load_guardian()
                    if guardian is None:
                        ctx.log.warn("Guardian model not available, skipping response filtering")
                        return

                    results = guardian.classify_all_risks(assistant_text)
                    blocked_categories = [
                        cat for cat, result in results.items()
                        if result["label"] == "Yes"
                    ]

                    if blocked_categories:
                        ctx.log.warn(f"Blocked assistant response due to: {blocked_categories}")
                        self.block_request(
                            flow,
                            category_name=", ".join(blocked_categories),
                            confidence=max(results[cat]["confidence"] for cat in blocked_categories),
                            matched_terms=blocked_categories
                        )

            except Exception as e:
                ctx.log.error(f"Error processing response: {str(e)}")

    def block_request(self, flow: http.HTTPFlow, category_name: str, confidence: float, matched_terms: list) -> None:
        response = {
            "error": {
                "message": f"The prompt was blocked because it contained content related to:"
                           f"\n {category_name}.",
                "type": "content_filter",
                "param": None,
                "code": "content_filter",
                "details": {
                    "confidence": round(confidence, 2),
                    "matched_terms": matched_terms[:10]
                }
            }
        }

        flow.response = http.Response.make(
            403,
            json.dumps(response),
            {"Content-Type": "application/json"}
        )

        ctx.log.warn(f"Blocked request: {response['error']['message']} (Confidence: {confidence:.2f})")

addons = [OpenAIProxy()]
