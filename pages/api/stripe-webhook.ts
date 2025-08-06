import Stripe from 'stripe'
import { buffer } from 'micro'
import { createClient } from '@supabase/supabase-js'
export const config = { api: { bodyParser: false } }
const stripe = new Stripe(process.env.STRIPE_SK!, { apiVersion: '2023-10-16' })
const db = ()=> createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE!)
export default async function handler(req:any,res:any){
  const sig = req.headers['stripe-signature'] as string
  let event: Stripe.Event
  try { event = stripe.webhooks.constructEvent(await buffer(req), sig, process.env.STRIPE_WEBHOOK!) }
  catch (err:any){ return res.status(400).send(`Webhook Error: ${err.message}`) }
  if (event.type === 'checkout.session.completed' && (event as any).livemode === true){
    const s = event.data.object as Stripe.Checkout.Session
    await db().from('jobs').insert({ email: s.customer_details?.email ?? null, status: 'paid' })
  }
  res.json({received:true})
}
