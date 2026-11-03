#!/usr/bin/env python3
"""
llamAmpere-Ultra: Automated Upstream Synchronizer
================================================
Monitors and synchronizes upstream developments from:
  1. JakeATX/llamAmpere (Core Ampere SM86 engine, SJ-KVaRN, MMA kernels)
  2. Niko1221/Strata (Memory tiering, APU/GPU offload research)
  3. AgentionAI (Gyro-Coder, PLE N-gram tables, Flash-Next models)

Preserves custom CUDA Driver VMM fast-swapping patches during updates.
"""

import os
import sys
import json
import subprocess
import urllib.request
import urllib.error

UPSTREAM_LLAMAMPERE = "https://github.com/JakeATX/llamAmpere.git"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_git(args: list[str], cwd: str = REPO_ROOT) -> tuple[int, str]:
    res = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return res.returncode, res.stdout.strip() + res.stderr.strip()

def check_github_releases(repo_slug: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo_slug}/releases"
    req = urllib.request.Request(url, headers={"User-Agent": "llamAmpere-Ultra-Sync"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data[:3]
    except Exception as e:
        return []

def check_hf_models(author: str) -> list[dict]:
    url = f"https://huggingface.co/api/models?author={author}"
    req = urllib.request.Request(url, headers={"User-Agent": "llamAmpere-Ultra-Sync"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data[:5]
    except Exception as e:
        return []

def main():
    print("=" * 75)
    print("  llamAmpere-Ultra: Upstream Synchronization & Ecosystem Monitor")
    print("=" * 75)

    # 1. Verify and configure git remotes
    code, remotes = run_git(["remote", "-v"])
    if "upstream" not in remotes:
        print(f">> Adding upstream remote: {UPSTREAM_LLAMAMPERE}")
        run_git(["remote", "add", "upstream", UPSTREAM_LLAMAMPERE])
    else:
        print(">> Upstream remote verified.")

    # 2. Fetch upstream llamAmpere
    print("\n[1/3] Fetching latest branches and tags from JakeATX/llamAmpere...")
    code, out = run_git(["fetch", "upstream", "--tags"])
    if code != 0:
        print(f"[ERROR] Failed fetching upstream: {out}")
    else:
        print(">> Upstream fetch successful.")

    code, current_branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    code, behind_count = run_git(["rev-list", "--count", f"HEAD..upstream/main"])
    code, ahead_count = run_git(["rev-list", "--count", f"upstream/main..HEAD"])

    print(f">> Active Local Branch: {current_branch}")
    print(f">> Status relative to upstream/main: Ahead: {ahead_count} commits | Behind: {behind_count} commits")

    code, latest_tag = run_git(["describe", "--tags", "--abbrev=0", "upstream/main"])
    print(f">> Latest upstream release tag: {latest_tag if code == 0 else 'Unknown'}")

    # 3. Check Strata releases
    print("\n[2/3] Checking Niko1221/Strata ecosystem releases...")
    strata_releases = check_github_releases("Niko1221/Strata")
    if strata_releases:
        latest_strata = strata_releases[0]
        print(f">> Latest Strata Release: {latest_strata.get('tag_name')} ({latest_strata.get('name')})")
        print(f"   Published at: {latest_strata.get('published_at')}")
    else:
        print(">> Could not query Strata releases (API rate limit or network).")

    # 4. Check AgentionAI models
    print("\n[3/3] Checking AgentionAI (Gyro-Coder / Flash-Next) Hugging Face models...")
    agention_models = check_hf_models("agentionai")
    if agention_models:
        for m in agention_models:
            print(f"   - {m.get('id')}")
    else:
        print(">> Could not query AgentionAI models.")

    print("\n" + "=" * 75)
    if int(behind_count) > 0:
        print(f"[ACTION AVAILABLE] {behind_count} new commits available in JakeATX/llamAmpere!")
        print("To safely integrate new upstream features while preserving CUDA VMM Swapping:")
        print("  1. git fetch upstream")
        print("  2. git merge upstream/main -m 'chore: sync upstream llamAmpere updates'")
        print("  3. Rebuild with cmake --build build-sm86 -j8 --target llama-server")
    else:
        print("[UP TO DATE] llamAmpere-Ultra is fully synchronized with upstream/main.")
    print("=" * 75)

if __name__ == "__main__":
    main()
