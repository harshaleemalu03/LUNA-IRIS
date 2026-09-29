#!/bin/sh
# Render the deployment-configurable API base into config.js, then hand over
# to the stock nginx entrypoint (it runs the image's /docker-entrypoint.d
# hooks and finally execs nginx with "$@").
#
# LUNA_API_BASE="" (the compose default) means "same origin": the UI calls
# /api/register on its own host and nginx proxies to the api service.
set -eu

envsubst '${LUNA_API_BASE}' \
  < /etc/luna-iris/config.js.template \
  > /usr/share/nginx/html/config.js

exec /docker-entrypoint.sh "$@"
