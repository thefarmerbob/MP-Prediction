import React, { useState } from 'react'
import './App.css'
import MicroplasticVisualizer from './components/MicroplasticVisualizer'
import EnhancedMicroplasticVisualizer from './components/EnhancedMicroplasticVisualizer'

function App() {
  const [useEnhanced, setUseEnhanced] = useState(true)

  return (
    <div className="App">
      <header className="App-header">
        <h1>Microplastic Time Series Visualizer</h1>
        <p>Interactive visualization of microplastic data over time</p>
        <div className="mode-toggle">
          <button 
            className={`toggle-btn ${!useEnhanced ? 'active' : ''}`}
            onClick={() => setUseEnhanced(false)}
          >
            Simple View
          </button>
          <button 
            className={`toggle-btn ${useEnhanced ? 'active' : ''}`}
            onClick={() => setUseEnhanced(true)}
          >
            Enhanced View
          </button>
        </div>
      </header>
      <main>
        {useEnhanced ? <EnhancedMicroplasticVisualizer /> : <MicroplasticVisualizer />}
      </main>
    </div>
  )
}

export default App
