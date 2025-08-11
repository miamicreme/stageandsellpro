# OFF
curl -sS -X POST https://YOUR-APP.modal.run/keepwarm_set \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"mode":"off"}'

# ALWAYS
curl -sS -X POST https://YOUR-APP.modal.run/keepwarm_set \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"mode":"always"}'

# BUSINESS hours (customize tz/hours/weekdays)
curl -sS -X POST https://YOUR-APP.modal.run/keepwarm_set \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"mode":"business","tz":"America/New_York","hours":"09:00-18:00","weekdays":"1-5"}'
