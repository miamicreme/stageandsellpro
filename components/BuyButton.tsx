export default function BuyButton({ priceId, label }:{ priceId:string; label:string }){
  const go = async ()=>{
    const r = await fetch('/api/checkout', { method:'POST', body: JSON.stringify({ priceId }) })
    const { url } = await r.json(); window.location.href = url
  }
  return <button onClick={go} className="bg-primary text-white px-6 py-3 rounded-card">{label}</button>
}
