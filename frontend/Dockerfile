# LUNA-IRiS frontend: static UI served by nginx with a same-origin proxy to
# the api service. Keeping /api and /outputs on this origin means the browser
# never crosses origins, so CORS (allowlist in backend/api/app.py) never
# applies to anything the Docker stack serves.
FROM nginx:1.29.8-alpine

# config.js is the one non-static file: it carries the deployment's API base
# URL and is re-rendered from the template by the entrypoint below. The
# selective COPY keeps this script/config/template out of the web root.
COPY index.html css js config.js /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY config.js.template /etc/luna-iris/config.js.template
COPY docker-entrypoint.sh /luna-docker-entrypoint.sh
RUN chmod +x /luna-docker-entrypoint.sh

# The base image's CMD (nginx -g 'daemon off;') is inherited; our entrypoint
# renders config.js first, then execs the stock nginx entrypoint so its own
# /docker-entrypoint.d hooks still run.
ENTRYPOINT ["/luna-docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
