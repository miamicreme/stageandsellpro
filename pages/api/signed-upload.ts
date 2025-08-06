import type { NextApiRequest, NextApiResponse } from 'next'
import { createClient } from '@supabase/supabase-js'
const admin = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE!)
export default async function handler(req:NextApiRequest,res:NextApiResponse){
  if (req.method!=='POST') return res.status(405).end()
  const { jobId, filename } = JSON.parse(req.body)
  const path = `jobs/${jobId}/${filename}`
  const { data, error } = await admin.storage.from('uploads').createSignedUploadUrl(path, 120)
  if (error) return res.status(400).json({ error: error.message })
  return res.status(200).json({ url: data.signedUrl, path })
}
