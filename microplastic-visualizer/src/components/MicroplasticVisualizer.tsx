import React, { useState, useRef, useEffect } from 'react';
import './MicroplasticVisualizer.css';

interface GifData {
  name: string;
  url: string;
  title: string;
  description: string;
}

const MicroplasticVisualizer: React.FC = () => {
  const [currentGif, setCurrentGif] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const imgRefs = useRef<(HTMLImageElement | null)[]>([null, null, null]);

  const gifs: GifData[] = [
    {
      name: 'microplastic_timeseries',
      url: '/gifs/microplastic_timeseries.gif',
      title: 'Full Microplastic Time Series',
      description: 'Complete view of microplastic concentration over time'
    },
    {
      name: 'microplastic_timeseries_cropped',
      url: '/gifs/microplastic_timeseries_cropped.gif',
      title: 'Cropped Microplastic Time Series',
      description: 'Focused regional view of microplastic concentration'
    },
    {
      name: 'clustering_red_timeseries',
      url: '/gifs/clustering_red_timeseries.gif',
      title: 'Clustering Analysis',
      description: 'Clustering visualization of microplastic distribution patterns'
    }
  ];

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    
    // Force GIF restart with new speed
    const img = imgRefs.current[currentGif];
    if (img) {
      const originalSrc = img.src;
      img.src = '';
      setTimeout(() => {
        img.src = originalSrc;
      }, 10);
    }
  };

  const handleGifSelect = (index: number) => {
    setCurrentGif(index);
  };

  // Force GIF restart by changing src
  const restartGif = () => {
    const img = imgRefs.current[currentGif];
    if (img) {
      const originalSrc = img.src;
      img.src = '';
      setTimeout(() => {
        img.src = originalSrc;
      }, 10);
    }
  };

  // Handle play/pause by manipulating the image
  useEffect(() => {
    const img = imgRefs.current[currentGif];
    if (img) {
      if (!isPlaying) {
        img.style.filter = 'grayscale(30%) brightness(0.7)';
        img.style.opacity = '0.6';
      } else {
        img.style.filter = 'none';
        img.style.opacity = '1';
      }
    }
  }, [isPlaying, currentGif]);

  // Handle speed change by manipulating CSS
  useEffect(() => {
    const img = imgRefs.current[currentGif];
    if (img) {
      // Remove existing dynamic style
      const existingStyle = document.getElementById('gif-speed-style');
      if (existingStyle) {
        existingStyle.remove();
      }
      
      // Add new dynamic style for speed control
      const style = document.createElement('style');
      style.id = 'gif-speed-style';
      style.textContent = `
        .gif-image-active {
          filter: ${!isPlaying ? 'grayscale(30%) brightness(0.7)' : 'none'};
          opacity: ${!isPlaying ? '0.6' : '1'};
          transform: scale(${0.95 + (playbackSpeed * 0.05)});
          transition: all 0.3s ease;
        }
        .speed-indicator::after {
          content: '${playbackSpeed}x';
          position: absolute;
          top: 10px;
          right: 10px;
          background: rgba(0, 123, 255, 0.8);
          color: white;
          padding: 5px 10px;
          border-radius: 15px;
          font-size: 12px;
          font-weight: bold;
        }
      `;
      document.head.appendChild(style);
      
      img.className = 'gif-image gif-image-active';
    }
  }, [playbackSpeed, currentGif, isPlaying]);

  return (
    <div className="microplastic-visualizer">
      <div className="gif-selector">
        {gifs.map((gif, index) => (
          <button
            key={gif.name}
            className={`gif-tab ${index === currentGif ? 'active' : ''}`}
            onClick={() => handleGifSelect(index)}
          >
            {gif.title}
          </button>
        ))}
      </div>

      <div className="gif-display">
        <div className="gif-container speed-indicator">
          {gifs.map((gif, index) => (
            <img
              key={gif.name}
              ref={(el) => (imgRefs.current[index] = el)}
              src={gif.url}
              alt={gif.title}
              className="gif-image"
              style={{
                display: index === currentGif ? 'block' : 'none',
                width: '100%',
                height: 'auto',
                maxHeight: '600px',
                objectFit: 'contain'
              }}
            />
          ))}
          <div className="gif-overlay">
            <h3>{gifs[currentGif].title}</h3>
            <p>{gifs[currentGif].description}</p>
          </div>
          {!isPlaying && (
            <div className="pause-indicator">
              <div className="pause-icon">⏸️</div>
            </div>
          )}
        </div>
      </div>

      <div className="controls-panel">
        <div className="playback-controls">
          <button className="control-btn" onClick={handlePlayPause}>
            {isPlaying ? '⏸️' : '▶️'}
          </button>
          <button className="control-btn" onClick={restartGif}>
            🔄
          </button>
        </div>

        <div className="speed-controls">
          <label htmlFor="speed-slider">Playback Speed: {playbackSpeed}x</label>
          <div className="speed-buttons">
            {[0.5, 1, 1.5, 2, 3].map((speed) => (
              <button
                key={speed}
                className={`speed-btn ${playbackSpeed === speed ? 'active' : ''}`}
                onClick={() => handleSpeedChange(speed)}
              >
                {speed}x
              </button>
            ))}
          </div>
          <input
            id="speed-slider"
            type="range"
            min="0.1"
            max="5"
            step="0.1"
            value={playbackSpeed}
            onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
            className="speed-slider"
          />
        </div>

        <div className="time-info">
          <div className="gif-info">
            <h4>Current Visualization:</h4>
            <p><strong>{gifs[currentGif].title}</strong></p>
            <p>{gifs[currentGif].description}</p>
            <div className="playback-info" style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px solid #dee2e6' }}>
              <p>Speed: {playbackSpeed}x</p>
              <p>Status: {isPlaying ? 'Playing' : 'Paused'}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MicroplasticVisualizer; 