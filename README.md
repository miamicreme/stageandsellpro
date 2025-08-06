# Stage \& Sell Pro (Explorer Install Pack)

Use **GitHub Desktop** + **Vercel** only (no terminal required).

1. Open GitHub Desktop → File → Add Local Repository… → select this folder → Publish to GitHub.
2. In Vercel → New Project → Import your repo → add env vars (see below) → Deploy.
3. Supabase (web): create project, run schema SQL, create buckets (uploads private, assets public).
4. Stripe (web): create 3 price IDs and a webhook to /api/stripe-webhook → copy signing secret to Vercel env.
5. Icons \& logos are already in /public (from your provided artwork).

## Environment Variables (set in Vercel → Settings → Environment Variables)

* NEXT\_PUBLIC\_SUPABASE\_URL
* NEXT\_PUBLIC\_SUPABASE\_ANON\_KEY
* SUPABASE\_SERVICE\_ROLE
* STRIPE\_SK
* STRIPE\_WEBHOOK
* NEXT\_PUBLIC\_STRIPE\_PRICE\_SINGLE
* NEXT\_PUBLIC\_STRIPE\_PRICE\_BUNDLE5
* NEXT\_PUBLIC\_STRIPE\_PRICE\_SUB10
* SITE\_URL (e.g. https://stageandsellpro.com)



&nbsp;Supabase (browser only)

Go to Supabase.com → New Project.



In SQL Editor, paste the schema:



create extension if not exists "uuid-ossp";

create table if not exists jobs (

&nbsp; id uuid primary key default gen\_random\_uuid(),

&nbsp; email text,

&nbsp; address text,

&nbsp; style text default 'modern',

&nbsp; files jsonb default '\[]',

&nbsp; assets jsonb default '{}',

&nbsp; status text not null default 'queued',

&nbsp; created\_at timestamptz default now()

);

create table if not exists revisions (

&nbsp; id uuid primary key default gen\_random\_uuid(),

&nbsp; job\_id uuid references jobs(id) on delete cascade,

&nbsp; notes text,

&nbsp; created\_at timestamptz default now()

);



In Storage, create buckets:



uploads (private)



assets (public read)



In Project Settings → API, copy:



Project URL → NEXT\_PUBLIC\_SUPABASE\_URL



anon public key → NEXT\_PUBLIC\_SUPABASE\_ANON\_KEY



service\_role key → SUPABASE\_SERVICE\_ROLE (server secret)





) Stripe (browser only)

In Stripe Dashboard → Products, create three prices:



Single Listing — $2,500 (one-time)



5-Listing Bundle — $11,000 (one-time)



Monthly Plan (10 listings) — $1,800 (recurring monthly)



Copy each Price ID into a note.



In Developers → Webhooks → Add endpoint, set URL to:

https://YOUR-VERCEL-DOMAIN/api/stripe-webhook

Select event checkout.session.completed and save.

Copy the signing secret → used as STRIPE\_WEBHOOK.





) Vercel (browser only)

Go to Vercel.com → New Project → Import Git Repository → choose your repo.



Set Environment Variables (Settings → Environment Variables) before the first deploy or set them after and redeploy:



NEXT\_PUBLIC\_SUPABASE\_URL (from Supabase)



NEXT\_PUBLIC\_SUPABASE\_ANON\_KEY (from Supabase)



SUPABASE\_SERVICE\_ROLE (server secret from Supabase)



STRIPE\_SK (Stripe Secret Key)



STRIPE\_WEBHOOK (signing secret from Stripe webhook)



NEXT\_PUBLIC\_STRIPE\_PRICE\_SINGLE (price id)



NEXT\_PUBLIC\_STRIPE\_PRICE\_BUNDLE5 (price id)



NEXT\_PUBLIC\_STRIPE\_PRICE\_SUB10 (price id)



SITE\_URL (e.g. https://stageandsellpro.com)



Click Deploy. Vercel detects Next.js and builds automatically.



Add your domain in Vercel → Project → Settings → Domains and point DNS per instructions there.



What you’ll get after deploy

Home page with your correct logo and branding.



/upload page that creates jobs and uploads photos directly to Supabase Storage via short-lived signed URLs (secure).



Stripe Checkout API route and webhook (server-side) to mark jobs as paid.



All icons (apple touch, Android 512, favicons) wired and in place.



Optional edits (GUI only)

Open the folder in VS Code (or any editor) to modify copy, colors, and images.



Save changes → go to GitHub Desktop → Commit to main → Push origin. Vercel auto-redeploys.



If you want, I can also produce a one-zip “Supabase SQL + screenshots” pack showing exactly where to click for each setting, or help you connect Modal later (which typically needs a small CLI step). 















