import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import Head from 'next/head'
export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <link rel="icon" href="/favicon-32x32.png" sizes="32x32"/>
        <link rel="icon" href="/favicon-16x16.png" sizes="16x16"/>
        <link rel="apple-touch-icon" href="/apple-touch-icon.png"/>
        <link rel="manifest" href="/site.webmanifest"/>
        <meta name="theme-color" content="#2563EB"/>
      </Head>
      <Component {...pageProps}/>
    </>
  )
}
