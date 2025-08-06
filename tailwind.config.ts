import type { Config } from 'tailwindcss'
const config: Config = {
  content: ['./pages/**/*.{ts,tsx}','./components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: { primary:'#2563EB', accent:'#059669', muted:'#F7FAFC', ink:'#111827' },
      borderRadius: { card: '1rem' },
      boxShadow: { card: '0 10px 30px rgba(2,6,23,0.08)' }
    }
  },
  plugins: []
}
export default config
