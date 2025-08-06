import type { NextApiRequest, NextApiResponse } from 'next'
import { createClient } from '@supabase/supabase-js'
const db = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE!)
export default async function handler(req:NextApiRequest,res:NextApiResponse){
  if (req.method==='POST'){
    const { address, style, email } = JSON.parse(req.body)
    const { data, error } = await db.from('jobs').insert({ address, style, email, status:'queued' }).select('id').single()
    if (error) return res.status(400).json({ error: error.message })
    return res.status(200).json({ id: data!.id })
  }
  if (req.method==='PUT'){
    const { id, files } = JSON.parse(req.body)
    const { error } = await db.from('jobs').update({ files }).eq('id', id)
    if (error) return res.status(400).json({ error: error.message })
    return res.status(200).json({ ok:true })
  }
  res.status(405).end()
}
