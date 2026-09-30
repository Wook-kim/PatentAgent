import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: './',
  plugins: [react()],
  define: {
    // 일부 의존성이 process.env 를 참조 (브라우저엔 없음)
    'process.env.NODE_ENV': JSON.stringify('production'),
    global: 'globalThis',
  },
  optimizeDeps: {
    // raphael(UMD/CJS)·ketcher 패키지를 esbuild 로 강제 사전번들링 → ESM 변환
    include: ['raphael', 'ketcher-core', 'ketcher-react', 'ketcher-standalone'],
    esbuildOptions: { target: 'es2020' },
  },
  build: {
    outDir: 'dist-embed',
    chunkSizeWarningLimit: 30000,
    commonjsOptions: {
      // ketcher-react dist(ESM)에 박힌 require() 호출까지 변환
      transformMixedEsModules: true,
    },
  },
})
