import React from 'react'
import { createRoot } from 'react-dom/client'
import { Editor } from 'ketcher-react'
import { StandaloneStructServiceProvider } from 'ketcher-standalone'
import 'ketcher-react/dist/index.css'

const structServiceProvider = new StandaloneStructServiceProvider()

// Ketcher 임베드: 부모 창과 postMessage로 통신
//   부모 -> iframe : {type:'ketcher:set', smiles} | {type:'ketcher:get', reqId}
//   iframe -> 부모 : {type:'ketcher:ready'} | {type:'ketcher:smiles', reqId, smiles, error}
let ketcherInstance = null

function onInit(ketcher) {
  ketcherInstance = ketcher
  window.ketcher = ketcher // 디버그용
  post({ type: 'ketcher:ready' })
}

function post(msg) {
  try { window.parent && window.parent.postMessage(msg, '*') } catch (e) {}
}

async function setSmiles(smiles) {
  if (!ketcherInstance) return
  try {
    if (smiles && String(smiles).trim()) {
      await ketcherInstance.setMolecule(String(smiles))
    } else {
      // 빈 입력이면 캔버스 비우기
      try { await ketcherInstance.setMolecule('') } catch (e) {}
    }
  } catch (e) {
    post({ type: 'ketcher:error', stage: 'set', error: String(e && e.message || e) })
  }
}

async function getSmiles(reqId) {
  if (!ketcherInstance) { post({ type: 'ketcher:smiles', reqId, smiles: '', error: 'not-ready' }); return }
  try {
    const smiles = await ketcherInstance.getSmiles()
    post({ type: 'ketcher:smiles', reqId, smiles })
  } catch (e) {
    post({ type: 'ketcher:smiles', reqId, smiles: '', error: String(e && e.message || e) })
  }
}

window.addEventListener('message', (ev) => {
  const d = ev.data || {}
  if (d.type === 'ketcher:set') setSmiles(d.smiles)
  else if (d.type === 'ketcher:get') getSmiles(d.reqId)
})

createRoot(document.getElementById('root')).render(
  <Editor
    staticResourcesUrl={import.meta.env.BASE_URL || './'}
    structServiceProvider={structServiceProvider}
    errorHandler={(m) => post({ type: 'ketcher:error', error: String(m) })}
    onInit={onInit}
  />
)
