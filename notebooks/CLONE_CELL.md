# Clone Cell

Paste this as the first cell in Colab or a VM notebook after the GitHub repository exists:

```python
GITHUB_REPO_URL = "https://github.com/QuangVoAI/surgical-video-assistant.git"
PROJECT_DIR = "/content/surgical-video-assistant"

import os, subprocess
from pathlib import Path

if not Path(PROJECT_DIR).exists():
    subprocess.run(["git", "clone", GITHUB_REPO_URL, PROJECT_DIR], check=True)
os.chdir(PROJECT_DIR)
print("Working directory:", os.getcwd())
```
