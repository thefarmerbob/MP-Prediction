import React, { useState, useRef, useEffect, useCallback } from 'react';
import './MicroplasticVisualizer.css';

interface GifData {
  name: string;
  url: string;
  title: string;
  description: string;
}

const EnhancedMicroplasticVisualizer: React.FC = () => {
  const [currentGif, setCurrentGif] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration] = useState<number>(60); // Estimated duration in seconds
  const imgRefs = useRef<(HTMLImageElement | null)[]>([null, null, null]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(Date.now());

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

  // Update time counter
  const updateTime = useCallback(() => {
    if (isPlaying) {
      setCurrentTime(prev => {
        const newTime = prev + (0.1 * playbackSpeed);
        return newTime >= duration ? 0 : newTime; // Loop back to start
      });
    }
  }, [isPlaying, playbackSpeed, duration]);

  // Start/stop time counter
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(updateTime, 100); // Update every 100ms
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, updateTime]);

  // Control GIF playback by manipulating the img element
  useEffect(() => {
    const img = imgRefs.current[currentGif];
    if (img) {
      // Force GIF restart to sync with our time
      const originalSrc = img.src;
      if (!isPlaying) {
        // Pause effect by reducing opacity and adding filter
        img.style.filter = 'grayscale(30%) brightness(0.7)';
        img.style.opacity = '0.6';
      } else {
        img.style.filter = 'none';
        img.style.opacity = '1';
        // Restart GIF to sync with time changes
        img.src = '';
        setTimeout(() => {
          img.src = originalSrc;
        }, 10);
      }
    }
  }, [isPlaying, currentGif, currentTime]);

  // Apply speed effect through CSS animation manipulation
  useEffect(() => {
    const img = imgRefs.current[currentGif];
    if (img) {
      // Use CSS transforms to simulate speed changes
      const speedClass = `speed-${playbackSpeed.toString().replace('.', '-')}`;
      img.className = `gif-image ${speedClass}`;
      
      // Add dynamic CSS for speed control
      const style = document.createElement('style');
      style.id = 'dynamic-speed-style';
      
      // Remove existing dynamic style
      const existingStyle = document.getElementById('dynamic-speed-style');
      if (existingStyle) {
        existingStyle.remove();
      }
      
      style.textContent = `
        .gif-image {
          animation-duration: ${8 / playbackSpeed}s;
          animation-timing-function: linear;
          animation-iteration-count: infinite;
        }
      `;
      document.head.appendChild(style);
    }
  }, [playbackSpeed, currentGif]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
    startTimeRef.current = Date.now() - (currentTime * 1000);
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    // Restart the GIF with new speed
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
    setCurrentTime(0);
    startTimeRef.current = Date.now();
  };

  const handleTimeChange = (time: number) => {
    setCurrentTime(time);
    startTimeRef.current = Date.now() - (time * 1000);
    
    // Force GIF restart at new time position
    const img = imgRefs.current[currentGif];
    if (img) {
      const originalSrc = img.src;
      img.src = '';
      setTimeout(() => {
        img.src = originalSrc;
      }, 10);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const getProgress = () => {
    return (currentTime / duration) * 100;
  };

  // Restart GIF function
  const restartGif = () => {
    setCurrentTime(0);
    startTimeRef.current = Date.now();
    const img = imgRefs.current[currentGif];
    if (img) {
      const originalSrc = img.src;
      img.src = '';
      setTimeout(() => {
        img.src = originalSrc;
      }, 10);
    }
  };

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
        <div className="gif-container">
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
                objectFit: 'contain',
                transition: 'all 0.3s ease'
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
        <div className="timeline-section">
          <div className="time-display">
            <span className="current-time">{formatTime(currentTime)}</span>
            <span className="duration">{formatTime(duration)}</span>
          </div>
          
          <div className="timeline-container">
            <div className="timeline-track">
              <div 
                className="timeline-progress" 
                style={{ width: `${getProgress()}%` }}
              />
              <input
                type="range"
                min="0"
                max={duration}
                value={currentTime}
                onChange={(e) => handleTimeChange(parseFloat(e.target.value))}
                className="timeline-slider"
                step="0.1"
              />
            </div>
          </div>
        </div>

        <div className="playback-controls">
          <button 
            className="control-btn"
            onClick={() => handleTimeChange(Math.max(0, currentTime - 10))}
            title="Skip back 10 seconds"
          >
            ⏪
          </button>
          <button 
            className="control-btn"
            onClick={() => handleTimeChange(Math.max(0, currentTime - 1))}
            title="Step back 1 second"
          >
            ⏮️
          </button>
          <button className="control-btn play-pause" onClick={handlePlayPause}>
            {isPlaying ? '⏸️' : '▶️'}
          </button>
          <button 
            className="control-btn"
            onClick={() => handleTimeChange(Math.min(duration, currentTime + 1))}
            title="Step forward 1 second"
          >
            ⏭️
          </button>
          <button 
            className="control-btn"
            onClick={() => handleTimeChange(Math.min(duration, currentTime + 10))}
            title="Skip forward 10 seconds"
          >
            ⏩
          </button>
          <button 
            className="control-btn"
            onClick={restartGif}
            title="Restart from beginning"
          >
            🔄
          </button>
        </div>

        <div className="speed-controls">
          <label htmlFor="speed-slider">Playback Speed: {playbackSpeed}x</label>
          <div className="speed-buttons">
            {[0.25, 0.5, 1, 1.5, 2, 3].map((speed) => (
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
            <div className="playback-info">
              <p>Time: {formatTime(currentTime)} / {formatTime(duration)}</p>
              <p>Speed: {playbackSpeed}x</p>
              <p>Status: {isPlaying ? 'Playing' : 'Paused'}</p>
              <p>Progress: {getProgress().toFixed(1)}%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedMicroplasticVisualizer; 