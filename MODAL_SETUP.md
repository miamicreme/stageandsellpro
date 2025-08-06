# Modal Auto-Render (GUI Setup)

This enables automated rendering without using a terminal by deploying the Modal worker via GitHub Actions.

## Step 1 — Create a Modal account & workspace
- Go to https://modal.com and sign up.
- Create a workspace (default is fine).

## Step 2 — Add a secret in Modal (GUI)
- In Modal → **Secrets** → **New Secret** named **`supabase-creds`**.
- Add two keys:
  - `SUPABASE_URL` = your Supabase project URL (from Supabase Settings → API)
  - `SUPABASE_KEY` = your **service role** key (server secret)

## Step 3 — Create an API token in Modal (GUI)
- In Modal → **Tokens** → **New token**.
- Copy **Token ID** and **Token Secret**.

## Step 4 — Add GitHub Action secrets (GUI)
- In your GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:
  - `MODAL_TOKEN_ID` = (paste from Modal)
  - `MODAL_TOKEN_SECRET` = (paste from Modal)

> You do **not** need to add Supabase secrets to GitHub. They live inside Modal as the `supabase-creds` secret you created in Step 2.

## Step 5 — Add these files to your repo (Explorer + GitHub Desktop)
- Copy the `modal/` folder and `.github/workflows/modal-deploy.yml` into your repo root.
- In GitHub Desktop: **Commit** → **Push**.

This push will trigger the **Modal Deploy** workflow. You can also trigger it manually in GitHub (Actions → Modal Deploy → Run workflow).

## Step 6 — Verify
- In GitHub → **Actions**, the job should finish green.
- In Modal dashboard → **Apps**, you should see **stage-sell-pro-pipeline** with a **schedule**.
- In Supabase → `jobs` table, create a test row (status `queued`, and `files` list of uploaded image paths). In 1–3 minutes, status flips to `processing` then `delivered`, and **assets** JSON contains public URLs from the `assets` bucket.

## Notes
- Current worker uses a lightweight Pillow enhancement as a placeholder. Swap in SDXL/ControlNet later.
- The function reads from `uploads` bucket and writes to `assets`. Ensure both buckets exist and `assets` is public.
- Costs: the current job runs on CPU and exits quickly. Switch to GPU by adding `gpu="A10G"` on `@stub.function` for `process_job` and including your model deps in the image.
