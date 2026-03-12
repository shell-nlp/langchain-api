#!/usr/bin/env python3
import json
import requests
from typing import Optional


API_URL = "http://192.168.102.19:8082/v1/images/generations"
DEFAULT_MODEL = "z_image"
DEFAULT_SIZE = "1664x928"


def generate_image(
    prompt: str,
    model: str = DEFAULT_MODEL,
    size: str = DEFAULT_SIZE,
    response_format: str = "url"
) -> Optional[str]:
    """
    Generate an image using the local image generation API.

    Args:
        prompt: Text description of the image to generate
        model: Model name (default: "z_image")
        size: Image dimensions (default: "1664x928")
        response_format: Response format, "url" or "b64_json" (default: "url")

    Returns:
        Image URL if successful, None otherwise
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "response_format": response_format,
        "size": size
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        if data.get("data") and len(data["data"]) > 0:
            if response_format == "url":
                return data["data"][0].get("url")
            elif response_format == "b64_json":
                return data["data"][0].get("b64_json")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error generating image: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using local API")
    parser.add_argument("prompt", help="Image description prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="Image size")
    parser.add_argument("--output", "-o", help="Output file path (optional)")

    args = parser.parse_args()

    url = generate_image(args.prompt, args.model, args.size)

    if url:
        print(json.dumps({"url": url}, indent=2))
    else:
        print(json.dumps({"error": "Failed to generate image"}, indent=2))
        exit(1)


if __name__ == "__main__":
    main()
