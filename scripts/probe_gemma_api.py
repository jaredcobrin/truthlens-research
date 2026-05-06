#!/usr/bin/env python3
"""Small Google AI Studio OpenAI-compatible API probe."""

from __future__ import annotations

import argparse
import getpass
import json
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--api-key-plain-stdin", action="store_true")
    args = parser.parse_args()

    if args.api_key_stdin:
        api_key = getpass.getpass("API key: ").strip()
    elif args.api_key_plain_stdin:
        api_key = input().strip()
    else:
        raise SystemExit("Use --api-key-stdin.")

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0,
        "max_tokens": 16,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "truthlens-benchmark/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            print(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}")
        print(exc.read().decode("utf-8", errors="replace"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
