import os
import base64
import logging

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB


def analyze_image_with_gemini(image_bytes: bytes, mime_type: str, question: str) -> str:
    """
    Analyze an image using the Google Gemini multimodal API.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        mime_type:   MIME type reported by the client (e.g. 'image/jpeg').
        question:    User's question about the image.

    Returns:
        The model's text answer.

    Raises:
        ValueError: If the API key is missing, the MIME type is unsupported,
                    or the image exceeds the size limit.
        RuntimeError: If the Gemini API call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please add it to your deployment environment."
        )

    # Normalize MIME type and validate
    normalized_mime = mime_type.split(";")[0].strip().lower()
    if normalized_mime not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported image type '{normalized_mime}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image size exceeds the 4 MB limit.")

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        image_part = {
            "inline_data": {
                "mime_type": normalized_mime,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
        }

        response = model.generate_content([image_part, question])
        return response.text

    except ValueError:
        raise
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise RuntimeError(f"Failed to get a response from Gemini: {exc}") from exc
