---
name: "image-generation"
description: "Generates images using a local API endpoint at 192.168.102.19:8082. Invoke when user wants to generate/create images, AI art, or illustrations from text prompts."
---

# Image Generation Skill

This skill generates images using a local image generation API via Python code execution.

## Quick Start

Run the image generation script directly:(这里必须是相对路径)

```bash
python skills/image-generation/scripts/generate_image.py "Your image description here"
```

### Command Line
(这里必须是相对路径)
```bash
python skills/image-generation/scripts/generate_image.py "身着红色汉服的中国女子" --size "1024x1024"
```

### Response Format

```json
{
  "url": "http://192.168.102.19:60203/static/a6f3468a-e582-43d8-8db5-fad60e4ca222.png"
}
```

## Notes

- Ensure the local API server is running at `192.168.102.19:8082` before making requests
- The generated image URL is accessible from the local network
- Image generation may take a few seconds depending on the complexity of the prompt
