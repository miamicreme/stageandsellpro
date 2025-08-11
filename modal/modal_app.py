name: Modal Deploy

on:
  workflow_dispatch:
  push:
    branches: [ main, master, develop ]
    paths:
      - "modal/**"
      - "modal_app.py"
      - ".github/workflows/modal-deploy.yml"

concurrency:
  group: modal-deploy-${{ github.workflow }}-${{ github.ref_name }}
  cancel-in-progress: true

permissions:
  contents: read

env:
  PYTHON_VERSION: "3.11"
  MODAL_ENTRYPOINT: "modal/modal_app.py"
  MODAL_PROFILE: "prod"

jobs:
  deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Modal CLI
        shell: bash
        run: |
          set -euo pipefail
          python -m pip install --upgrade pip
          pip install 'modal>=1.1,<2'
          python -m modal --version || true

      - name: Configure Modal token
        shell: bash
        env:
          CI_MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
          CI_MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
          MODAL_PROFILE: prod
        run: |
          set -euo pipefail

          if [ -z "${CI_MODAL_TOKEN_ID:-}" ] || [ -z "${CI_MODAL_TOKEN_SECRET:-}" ]; then
            echo "::error::Missing MODAL_TOKEN_ID or MODAL_TOKEN_SECRET in repo secrets."
            exit 1
          fi

          # Write creds into the prod profile
          python -m modal token set \
            --token-id "$CI_MODAL_TOKEN_ID" \
            --token-secret "$CI_MODAL_TOKEN_SECRET" \
            --profile "$MODAL_PROFILE"

          # Verify ~/.modal.toml has [prod] with token_id + token_secret
          python -c 'import sys, pathlib, tomllib; p=pathlib.Path.home()/".modal.toml"; d=tomllib.loads(p.read_text()) if p.exists() else {}; pr=d.get("prod") or {}; sys.exit(0 if (pr.get("token_id") and pr.get("token_secret")) else 2)' \
          || { echo "::error::Modal token verification failed (missing [prod].token_id/token_secret in ~/.modal.toml)"; exit 1; }

          echo "MODAL_PROFILE=$MODAL_PROFILE" >> "$GITHUB_ENV"
          unset CI_MODAL_TOKEN_ID CI_MODAL_TOKEN_SECRET

      - name: Validate entrypoint exists
        shell: bash
        run: |
          set -euo pipefail
          if [ ! -f "$MODAL_ENTRYPOINT" ]; then
            echo "::error file=$MODAL_ENTRYPOINT::Modal entrypoint not found"
            exit 1
          fi

      - name: Deploy to Modal
        shell: bash
        env:
          MODAL_PROFILE: ${{ env.MODAL_PROFILE }}
          MODAL_ENTRYPOINT: ${{ env.MODAL_ENTRYPOINT }}
        run: |
          set -euo pipefail
          echo "Using MODAL_PROFILE=${MODAL_PROFILE}"
          echo "Deploying entrypoint: ${MODAL_ENTRYPOINT}"

          for attempt in 1 2 3; do
            if env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET \
              python -m modal deploy "${MODAL_ENTRYPOINT}"; then
              echo "Deploy succeeded."
              exit 0
            fi
            echo "Deploy failed (attempt ${attempt}). Retrying in 10s..."
            sleep 10
          done

          echo "::error::Modal deploy failed after 3 attempts."
          exit 1

      # ================== Post-deploy: configure keep-warm mode ==================
      - name: Configure keep-warm mode
        if: success()
        shell: bash
        env:
          # REQUIRED: your Modal app prefix (before -<function>.modal.run)
          # e.g., miamicreme--stage-sell-pro-pipeline
          SSP_APP_PREFIX: ${{ secrets.SSP_APP_PREFIX }}

          # REQUIRED: API key configured in the app (for keepwarm_set auth)
          SSP_API_KEY: ${{ secrets.STAGESELLPRO_KEY }}

          # Optional: override; if empty we auto-pick by branch (main=business, others=off)
          KEEPWARM_MODE: "off"                 # off | business | always
          KEEPWARM_TZ: "America/New_York"   # for business mode
          KEEPWARM_HOURS: "09:00-18:00"     # for business mode
          KEEPWARM_WEEKDAYS: "1-5"          # for business mode
        run: |
          set -euo pipefail

          if [ -z "${SSP_APP_PREFIX:-}" ] || [ -z "${SSP_API_KEY:-}" ]; then
            echo "::warning::SSP_APP_PREFIX or STAGESELLPRO_KEY secret missing; skipping keep-warm config."
            exit 0
          fi

          MODE="${KEEPWARM_MODE:-}"
          if [ -z "$MODE" ]; then
            case "${GITHUB_REF_NAME}" in
              main|master) MODE="business" ;;
              develop|dev) MODE="off" ;;
              *) MODE="off" ;;
            esac
          fi

          URL="https://${SSP_APP_PREFIX}-keepwarm_set.modal.run/"
          echo "Setting keep-warm mode='${MODE}' at ${URL}"

          if [ "$MODE" = "business" ]; then
            BODY=$(printf '{"mode":"%s","tz":"%s","hours":"%s","weekdays":"%s"}' \
              "$MODE" "$KEEPWARM_TZ" "$KEEPWARM_HOURS" "$KEEPWARM_WEEKDAYS")
          else
            BODY=$(printf '{"mode":"%s"}' "$MODE")
          fi

          curl -fSs -X POST "$URL" \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${SSP_API_KEY}" \
            -d "$BODY" \
            -o /tmp/keepwarm_set.json

          echo "Keep-warm set response:"
          cat /tmp/keepwarm_set.json

      - name: Verify keep-warm mode
        if: success()
        shell: bash
        env:
          SSP_APP_PREFIX: ${{ secrets.SSP_APP_PREFIX }}
        run: |
          set -euo pipefail
          if [ -z "${SSP_APP_PREFIX:-}" ]; then
            echo "::warning::SSP_APP_PREFIX secret missing; skipping verification."
            exit 0
          fi
          URL="https://${SSP_APP_PREFIX}-keepwarm_status.modal.run/"
          echo "Verifying keep-warm at ${URL}"
          curl -fSs "$URL" -o /tmp/keepwarm_status.json
          echo "Status:"
          cat /tmp/keepwarm_status.json
