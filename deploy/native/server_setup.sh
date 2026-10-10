#!/usr/bin/env bash
# Run ONCE, with sudo, on the server that will host this. Everything after this is `git push` —
# no Docker, no subscription, no further root needed. Creates one unprivileged account that owns
# the bare repo, the work tree, and its own user-level systemd units.
set -euo pipefail
: "${DEPLOY_USER:=oncosync}"

if ! id -u "$DEPLOY_USER" &>/dev/null; then
  echo "[server_setup] creating user $DEPLOY_USER"
  useradd -m -s /bin/bash "$DEPLOY_USER"
fi

# Lets the user's systemd session (and its units) keep running with no one logged in — required for
# `systemctl --user` to work from a non-interactive git-push/ssh session. This is the one genuinely
# easy-to-miss step; without it the post-receive hook's `systemctl --user restart` silently fails.
loginctl enable-linger "$DEPLOY_USER"

HOME_DIR="$(getent passwd "$DEPLOY_USER" | cut -d: -f6)"
BARE_REPO="$HOME_DIR/oncosync-deploy.git"
WORK_TREE="$HOME_DIR/oncosync"

echo "[server_setup] python3 / awscli (apt — adjust for your distro)"
apt-get update -qq && apt-get install -y -qq python3 python3-venv awscli >/dev/null || \
  echo "[server_setup] WARN: apt install failed/skipped — ensure python3, python3-venv, aws cli are present"

sudo -u "$DEPLOY_USER" mkdir -p "$BARE_REPO" "$WORK_TREE" "$HOME_DIR/.config/oncosync" "$HOME_DIR/.config/systemd/user"
sudo -u "$DEPLOY_USER" git init --bare "$BARE_REPO"

HOOK_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/post-receive"
sudo -u "$DEPLOY_USER" cp "$HOOK_SRC" "$BARE_REPO/hooks/post-receive"
sudo -u "$DEPLOY_USER" chmod +x "$BARE_REPO/hooks/post-receive"

UNIT_DIR="$HOME_DIR/.config/systemd/user"
for f in oncosync-api.service oncosync-s3-sync-up.service oncosync-s3-sync-up.timer; do
  sudo -u "$DEPLOY_USER" cp "$(dirname "${BASH_SOURCE[0]}")/$f" "$UNIT_DIR/$f"
done

if [ ! -f "$HOME_DIR/.config/oncosync/env" ]; then
  sudo -u "$DEPLOY_USER" bash -c "cat > '$HOME_DIR/.config/oncosync/env' <<'EOF'
# Fill in to enable S3; leave as-is (or delete this file) to run local-only.
# BC_S3_BUCKET=your-bucket
# BC_S3_PREFIX=is360-breast-poc
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_DEFAULT_REGION=ap-south-1
EOF"
  chmod 600 "$HOME_DIR/.config/oncosync/env"
  chown "$DEPLOY_USER":"$DEPLOY_USER" "$HOME_DIR/.config/oncosync/env"
fi

sudo -u "$DEPLOY_USER" env XDG_RUNTIME_DIR="/run/user/$(id -u "$DEPLOY_USER")" systemctl --user daemon-reload
sudo -u "$DEPLOY_USER" env XDG_RUNTIME_DIR="/run/user/$(id -u "$DEPLOY_USER")" systemctl --user enable oncosync-s3-sync-up.timer
sudo -u "$DEPLOY_USER" env XDG_RUNTIME_DIR="/run/user/$(id -u "$DEPLOY_USER")" systemctl --user start oncosync-s3-sync-up.timer

echo
echo "[server_setup] done."
echo "On your laptop, inside the IS360_Breast_PoC repo:"
echo "  git remote add deploy ${DEPLOY_USER}@$(hostname -f 2>/dev/null || hostname):${BARE_REPO}"
echo "  git push deploy main"
echo "First push will be empty of workspace data (by design — see .gitignore); run an experiment or"
echo "point S3 sync-down at it afterward to populate ~oncosync/oncosync/workspace."
