from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://api.github.com"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a GitHub repository and push the current local repo."
    )
    parser.add_argument("--repo", default="surgical-video-assistant", help="GitHub repository name.")
    parser.add_argument("--private", action="store_true", help="Create a private repository.")
    parser.add_argument("--description", default="Surgical frame VQA and phase/tool recognition assistant.")
    parser.add_argument(
        "--remote-url",
        help=(
            "Existing empty GitHub repository URL. Use this when the token cannot create repos, "
            "for example https://github.com/USER/surgical-video-assistant.git"
        ),
    )
    parser.add_argument("--skip-create", action="store_true", help="Do not call the GitHub create-repo API.")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit(
            "Missing GITHUB_TOKEN. Create a new GitHub token, then run:\n"
            "  export GITHUB_TOKEN='paste-token-here'\n"
            "Do not commit or paste the token into notebooks/chat."
        )

    user = github_request("GET", "/user", token)
    username = user["login"]

    if args.remote_url:
        clone_url = args.remote_url
        full_name = clone_url.removesuffix(".git").split("github.com/")[-1]
    elif args.skip_create:
        clone_url = f"https://github.com/{username}/{args.repo}.git"
        full_name = f"{username}/{args.repo}"
    else:
        try:
            repo = create_or_get_repo(token, args.repo, args.description, private=args.private)
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                raise SystemExit(
                    "GitHub returned 403 while creating the repository.\n"
                    "Most likely the token cannot create repos.\n\n"
                    "Fix option A: create an empty repo on GitHub named "
                    f"'{args.repo}', then run:\n"
                    f"  python scripts/publish_to_github.py --skip-create --repo {args.repo}\n\n"
                    "Fix option B: create a new token with permission to create repositories."
                ) from exc
            raise
        clone_url = repo["clone_url"]
        full_name = repo["full_name"]

    run(["git", "remote", "remove", "origin"], allow_failure=True)
    run(["git", "remote", "add", "origin", clone_url])
    run(["git", "branch", "-M", "main"])
    git_push_with_token(token)

    print(f"Pushed to https://github.com/{full_name}")
    print("Notebook clone URL:")
    print(f"https://github.com/{username}/{args.repo}.git")


def create_or_get_repo(token: str, name: str, description: str, private: bool) -> dict:
    try:
        return github_request("POST", "/user/repos", token, {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": False,
        })
    except urllib.error.HTTPError as exc:
        if exc.code != 422:
            raise
        user = github_request("GET", "/user", token)
        return github_request("GET", f"/repos/{user['login']}/{name}", token)


def github_request(method: str, path: str, token: str, payload: dict | None = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL + path,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def run(command: list[str], allow_failure: bool = False) -> None:
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
    if result.returncode and not allow_failure:
        raise SystemExit(result.returncode)


def git_push_with_token(token: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        askpass = Path(tmp_dir) / "git_askpass.py"
        askpass.write_text(
            "import os, sys\n"
            "prompt = sys.argv[1].lower() if len(sys.argv) > 1 else ''\n"
            "if 'username' in prompt:\n"
            "    print('x-access-token')\n"
            "else:\n"
            "    print(os.environ['GITHUB_TOKEN'])\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["GIT_ASKPASS"] = f"{sys.executable} {askpass}"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GITHUB_TOKEN"] = token
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
        )
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
