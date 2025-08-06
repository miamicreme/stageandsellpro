# Stage & Sell Pro (Explorer Install Pack)

Use **GitHub Desktop** + **Vercel** only (no terminal required).

1) Open GitHub Desktop → File → Add Local Repository… → select this folder → Publish to GitHub.
2) In Vercel → New Project → Import your repo → add env vars (see below) → Deploy.
3) Supabase (web): create project, run schema SQL, create buckets (uploads private, assets public).
4) Stripe (web): create 3 price IDs and a webhook to /api/stripe-webhook → copy signing secret to Vercel env.
5) Icons & logos are already in /public (from your provided artwork).

## Environment Variables (set in Vercel → Settings → Environment Variables)
- NEXT_PUBLIC_SUPABASE_URL
- NEXT_PUBLIC_SUPABASE_ANON_KEY
- SUPABASE_SERVICE_ROLE
- STRIPE_SK
- STRIPE_WEBHOOK
- NEXT_PUBLIC_STRIPE_PRICE_SINGLE
- NEXT_PUBLIC_STRIPE_PRICE_BUNDLE5
- NEXT_PUBLIC_STRIPE_PRICE_SUB10
- SITE_URL (e.g. https://stageandsellpro.com)
