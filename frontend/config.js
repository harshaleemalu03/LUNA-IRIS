/* API base URL read by js/app.js (window.LUNA_API_BASE).
   Checked-in default: the hosted HF Space backend, so a plain static
   deployment (`python3 -m http.server 5173`) behaves exactly as it does
   today. In the Docker image this file is re-rendered at container start
   from config.js.template, where LUNA_API_BASE defaults to "" (same-origin
   — the UI talks to the nginx /api proxy).
   "" = same origin; any absolute URL is used as-is. */
window.LUNA_API_BASE = 'https://harshaleemalu03-luna-iris.hf.space';
