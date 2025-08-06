export default function BrandHeader(){
  return (
    <header className="w-full border-b border-slate-200 bg-white">
      <div className="mx-auto max-w-6xl px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img src="/logo-light.svg" alt="Stage & Sell Pro" className="h-8 w-auto"/>
          <span className="sr-only">Stage & Sell Pro</span>
        </div>
        <nav className="flex items-center gap-6 text-sm">
          <a className="text-ink hover:text-primary" href="#pricing">Pricing</a>
          <a className="bg-primary text-white px-4 py-2 rounded-card" href="/upload">Get Started</a>
        </nav>
      </div>
    </header>
  )
}
