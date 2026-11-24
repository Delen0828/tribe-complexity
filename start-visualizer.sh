#!/usr/bin/env bash
# Install and start the visualizer and its dedicated Cloudflare Tunnel as user services.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
hostname=tribe.songwen.dev
tunnel_name=tribe-visualizer
port=5501
python_args=()

usage() {
    cat <<'HELP'
Usage: ./start-visualizer.sh [--port PORT] [--python PATH]

Publish the visualizer at https://tribe.songwen.dev/visualizer/ using Cloudflare
Tunnel. The local origin listens on 127.0.0.1:5501 by default. Both user services
stay running after logout and start after reboot.

Requires cloudflared, systemd user services, and access to the songwen.dev
Cloudflare zone. Run ./start-visualizer-dev.sh for local development.
HELP
}

while (($#)); do
    case "$1" in
        --port|--python)
            if (($# < 2)) || [[ "$2" == --* ]]; then
                printf 'Missing value for %s.\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == --port ]]; then
                port="$2"
            else
                python_path="$(realpath -s -- "$2")"
                if [[ ! -x "$python_path" ]]; then
                    printf 'Python is not executable at %s.\n' "$python_path" >&2
                    exit 2
                fi
                python_args=(--python "$python_path")
            fi
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || ((port > 65535)); then
    echo 'Port must be between 1 and 65535.' >&2
    exit 2
fi

cloudflared_bin="$(command -v cloudflared)" || { echo 'Install cloudflared first.' >&2; exit 1; }
config_dir="$HOME/.cloudflared"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
config="$config_dir/tribe-visualizer.yml"
cert="$config_dir/cert.pem"
origin_service=tribe-visualizer.service
tunnel_service=cloudflared-tribe.service

systemctl --user show-environment >/dev/null
if [[ "$(loginctl show-user "$(id -un)" -p Linger --value)" != yes ]]; then
    loginctl enable-linger "$(id -un)"
fi

umask 077
mkdir -p -- "$config_dir" "$unit_dir"
exec 9>"$config_dir/tribe-visualizer.lock"
flock 9

if [[ ! -r "$cert" ]]; then
    "$cloudflared_bin" tunnel login
fi
cloudflare_tunnel() {
    "$cloudflared_bin" tunnel --origincert "$cert" "$@"
}
find_tunnel() {
    cloudflare_tunnel list --name "$tunnel_name" --output json | python3 -c '
import json, sys
matches = [entry["id"] for entry in json.load(sys.stdin) or [] if entry["name"] == sys.argv[1]]
if len(matches) > 1:
    sys.exit("More than one Cloudflare tunnel matches the deployment name.")
if matches:
    print(matches[0])
' "$tunnel_name"
}

tunnel_id="$(find_tunnel)"
if [[ -z "$tunnel_id" ]]; then
    cloudflare_tunnel create "$tunnel_name"
    tunnel_id="$(find_tunnel)"
fi
credentials="$config_dir/$tunnel_id.json"
if [[ -z "$tunnel_id" || ! -r "$credentials" ]]; then
    echo 'The dedicated tunnel credentials are missing from ~/.cloudflared.' >&2
    exit 1
fi

work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT

systemd_arg() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '"%s"' "$value"
}

python3 - "$tunnel_id" "$credentials" "$hostname" "$port" > "$work_dir/tunnel.yml" <<'PY'
import json
import sys

tunnel_id, credentials, hostname, port = sys.argv[1:]
print(json.dumps({
    "tunnel": tunnel_id,
    "credentials-file": credentials,
    "ingress": [
        {"hostname": hostname, "service": f"http://127.0.0.1:{port}"},
        {"service": "http_status:404"},
    ],
}, indent=2))
PY
"$cloudflared_bin" tunnel --config "$work_dir/tunnel.yml" ingress validate

cat > "$work_dir/$origin_service" <<EOF_ORIGIN
[Unit]
Description=TRIBE visualizer

[Service]
ExecStart=:/usr/bin/bash $(systemd_arg "$repo_root/start-visualizer-dev.sh") --host 127.0.0.1 --port $port --public --no-open$(for arg in "${python_args[@]}"; do printf ' '; systemd_arg "$arg"; done)
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=default.target
EOF_ORIGIN
cat > "$work_dir/$tunnel_service" <<EOF_TUNNEL
[Unit]
Description=Cloudflare Tunnel for $hostname
BindsTo=$origin_service
After=$origin_service

[Service]
ExecStart=:$(systemd_arg "$cloudflared_bin") --no-autoupdate tunnel --config $(systemd_arg "$config") run
Restart=always
RestartSec=5s

[Install]
WantedBy=default.target $origin_service
EOF_TUNNEL

install -m 0600 -- "$work_dir/tunnel.yml" "$config"
install -m 0644 -- "$work_dir/$origin_service" "$work_dir/$tunnel_service" "$unit_dir/"
systemctl --user daemon-reload
systemctl --user enable "$origin_service"
systemctl --user restart "$origin_service"
curl --fail --silent --show-error --retry 10 --retry-delay 1 --retry-connrefused --max-time 4 \
    "http://127.0.0.1:$port/visualizer/" >/dev/null

cloudflare_tunnel route dns "$tunnel_id" "$hostname"
systemctl --user enable "$tunnel_service"
systemctl --user restart "$tunnel_service"
public_url="https://$hostname/visualizer/"
if ! curl --fail --silent --location --max-time 5 "$public_url" >/dev/null; then
    # Local resolvers can cache NXDOMAIN from before the route was created.
    curl --fail --silent --show-error --location --retry 12 --retry-delay 2 --retry-all-errors --max-time 5 \
        --doh-url https://cloudflare-dns.com/dns-query "$public_url" >/dev/null
fi
printf 'Visualizer: %s\n' "$public_url"
