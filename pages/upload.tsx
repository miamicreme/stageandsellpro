import { useState } from 'react'
const STYLES = ['modern','industrial','contemporary']
export default function Upload(){
  const [address,setAddress]=useState(''); const [style,setStyle]=useState(STYLES[0]);
  const [files,setFiles]=useState<File[]>([]); const [busy,setBusy]=useState(false);
  const submit = async ()=>{
    setBusy(true)
    const job = await fetch('/api/create-job',{method:'POST',body:JSON.stringify({address,style})}).then(r=>r.json())
    const paths:string[] = []
    for (const f of files){
      const s = await fetch('/api/signed-upload',{method:'POST',body:JSON.stringify({jobId:job.id, filename:f.name})}).then(r=>r.json())
      await fetch(s.url,{ method:'PUT', headers:{'x-upsert':'true'}, body:f })
      paths.push(s.path)
    }
    await fetch('/api/create-job',{method:'PUT',body:JSON.stringify({id:job.id, files:paths})})
    alert('Uploads received. Processing will begin shortly.'); setBusy(false)
  }
  return (
    <div className="max-w-3xl mx-auto p-8">
      <h1 className="text-3xl font-bold text-primary mb-4">Upload Photos & Floor Plan</h1>
      <input className="w-full border p-3 rounded-card mb-3" placeholder="Property address" value={address} onChange={e=>setAddress(e.target.value)}/>
      <select className="w-full border p-3 rounded-card mb-3" value={style} onChange={e=>setStyle(e.target.value)}>
        {STYLES.map(s=> <option key={s} value={s}>{s}</option>)}
      </select>
      <input type="file" multiple accept="image/*" onChange={e=>setFiles(Array.from(e.target.files||[]))} className="w-full border p-3 rounded-card mb-3"/>
      <button disabled={!address||!files.length||busy} onClick={submit} className="bg-[#059669] text-white px-6 py-3 rounded-card">{busy?'Uploading…':'Submit'}</button>
    </div>
  )
}
