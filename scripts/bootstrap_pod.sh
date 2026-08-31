#!/usr/bin/env bash
# Bootstrap GitHub CLI authentication in a fresh Debian/Ubuntu GPU Pod.

set -euo pipefail

run_as_root() {
    # Run package-manager commands directly as root or through sudo otherwise.
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "ERROR: root privileges or sudo are required to install GitHub CLI." >&2
        return 1
    fi
}

if ! command -v gh >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "ERROR: automatic gh installation currently supports Debian/Ubuntu Pods only." >&2
        exit 1
    fi

    echo "GitHub CLI is missing; installing it with apt-get..."
    run_as_root apt-get update
    run_as_root apt-get install -y gh
else
    echo "GitHub CLI is already installed: $(gh --version | head -n 1)"
fi

# Configure the commit identity used by fresh Pods.
git config --global user.email "sz.yixinzou@gmail.com"
git config --global user.name "Yixin Zou"

if ! gh auth status --hostname github.com >/dev/null 2>&1; then
    # Device login keeps credentials out of the repository and shell history.
    echo "GitHub authentication is required. Complete the device flow in your browser."
    gh auth login --hostname github.com --git-protocol https --web
fi

# Configure Git to ask gh for HTTPS credentials instead of prompting for a password.
gh auth setup-git
gh auth status --hostname github.com

echo "GitHub CLI and Git authentication are ready."
