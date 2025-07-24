import React, { useRef, useEffect, useState, useCallback } from 'react';

interface FrameControlledGifProps {
  src: string;
  alt: string;
  className?: string;
  isPlaying: boolean;
  playbackSpeed: number;
  onFrameChange?: (frame: number, totalFrames: number) => void;
}

const FrameControlledGif: React.FC<FrameControlledGifProps> = ({
  src,
  alt,
  className,
  isPlaying,
  playbackSpeed,
  onFrameChange
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const animationRef = useRef<number>(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [frames, setFrames] = useState<ImageData[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);
  const lastFrameTime = useRef<number>(0);

  // Extract frames from GIF (simplified approach)
  const extractFrames = useCallback(async () => {
    if (!imgRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    try {
      // For now, we'll use the original image as a single frame
      // In a real implementation, you'd need a library like 'gif-frames' or 'omggif'
      const img = imgRef.current;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      
      ctx.drawImage(img, 0, 0);
      const frameData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      setFrames([frameData]);
      setTotalFrames(1);
      setIsLoaded(true);
    } catch (error) {
      console.error('Error extracting frames:', error);
    }
  }, []);

  // Animation loop
  const animate = useCallback((timestamp: number) => {
    if (!isPlaying || frames.length === 0) return;

    const frameDelay = 100 / playbackSpeed; // Adjust based on playback speed
    
    if (timestamp - lastFrameTime.current >= frameDelay) {
      setCurrentFrame(prev => {
        const nextFrame = (prev + 1) % frames.length;
        onFrameChange?.(nextFrame, frames.length);
        return nextFrame;
      });
      lastFrameTime.current = timestamp;
    }

    animationRef.current = requestAnimationFrame(animate);
  }, [isPlaying, playbackSpeed, frames.length, onFrameChange]);

  // Draw current frame
  useEffect(() => {
    if (!canvasRef.current || !frames[currentFrame]) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.putImageData(frames[currentFrame], 0, 0);
  }, [currentFrame, frames]);

  // Start/stop animation
  useEffect(() => {
    if (isPlaying && frames.length > 0) {
      animationRef.current = requestAnimationFrame(animate);
    } else {
      cancelAnimationFrame(animationRef.current);
    }

    return () => cancelAnimationFrame(animationRef.current);
  }, [isPlaying, animate, frames.length]);

  // Load image and extract frames
  useEffect(() => {
    if (imgRef.current) {
      imgRef.current.onload = extractFrames;
      imgRef.current.src = src;
    }
  }, [src, extractFrames]);

  return (
    <div className={className}>
      <img
        ref={imgRef}
        style={{ display: 'none' }}
        alt={alt}
        crossOrigin="anonymous"
      />
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: 'auto',
          maxHeight: '600px',
          objectFit: 'contain'
        }}
      />
      {!isLoaded && (
        <div style={{ 
          position: 'absolute', 
          top: '50%', 
          left: '50%', 
          transform: 'translate(-50%, -50%)',
          color: '#666'
        }}>
          Loading frames...
        </div>
      )}
    </div>
  );
};

export default FrameControlledGif; 