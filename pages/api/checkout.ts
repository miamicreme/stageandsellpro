import type { NextApiRequest, NextApiResponse } from 'next'
import Stripe from 'stripe'
const stripe = new Stripe(process.env.STRIPE_SK!, { apiVersion: '2023-10-16' })
export default async function handler(req:NextApiRequest,res:NextApiResponse){
  const { priceId } = JSON.parse(req.body)
  const session = await stripe.checkout.sessions.create({
    mode:'payment',
    line_items:[{ price: priceId, quantity:1 }],
    success_url: `${process.env.SITE_URL}/success?sid={CHECKOUT_SESSION_ID}`,
    cancel_url: `${process.env.SITE_URL}/#pricing`
  })
  res.status(200).json({ url: session.url })
}
