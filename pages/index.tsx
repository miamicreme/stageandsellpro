import BrandHeader from '@/components/BrandHeader'
export default function Home(){
  return (
    <>
      <BrandHeader/>
      <main className="mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-4xl md:text-5xl font-bold text-ink">
          Lift your listing value in <span className="text-primary">48 hours</span>.
        </h1>
        <p className="mt-4 text-slate-600 max-w-2xl">
          AI-powered virtual staging + marketing assets by Miami Creme Labs.
        </p>
        <div className="mt-8 flex gap-4">
          <a className="bg-[#059669] text-white px-6 py-3 rounded-card" href="/upload">Start now</a>
          <a className="px-6 py-3 rounded-card border border-slate-300" href="#pricing">See pricing</a>
        </div>
      </main>
    </>
  )
}
