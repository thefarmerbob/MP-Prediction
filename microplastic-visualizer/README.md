# Microplastic Time Series Visualizer

A React-based interactive visualization tool for exploring microplastic concentration data over time using three different GIF animations.

## Features

### 🎯 **Dual View Modes**
- **Simple View**: Basic GIF display with play/pause and speed controls
- **Enhanced View**: Advanced controls with timeline scrubbing and frame navigation

### 🎮 **Interactive Controls**

#### Timeline Controls (Enhanced View)
- **Time Scrubber**: Drag to jump to any point in the timeline
- **Time Display**: Shows current time and total duration
- **Progress Bar**: Visual indicator of playback progress

#### Playback Controls
- **Play/Pause**: Start or stop the animation
- **Frame Navigation**: Step forward/backward by individual frames or 10-second jumps
- **Restart**: Reset animation to the beginning
- **Speed Control**: Adjust playback speed from 0.1x to 5x

#### Speed Options
- **Quick Buttons**: 0.25x, 0.5x, 1x, 1.5x, 2x, 3x
- **Slider**: Fine-tune speed with precision control
- **Real-time Adjustment**: Change speed while playing

### 📊 **Three Visualization Types**

1. **Full Microplastic Time Series**
   - Complete global view of microplastic concentration
   - Shows comprehensive temporal patterns

2. **Cropped Microplastic Time Series** 
   - Focused regional analysis
   - Detailed view of specific geographic areas

3. **Clustering Analysis**
   - Advanced clustering visualization
   - Reveals distribution patterns and hotspots

### 🎨 **Modern UI Features**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Smooth Animations**: Fluid transitions and hover effects
- **Visual Feedback**: Pause indicators and loading states
- **Information Overlays**: Descriptive text for each visualization

## Getting Started

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd microplastic-visualizer
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   - Navigate to `http://localhost:5173`

### File Structure
```
microplastic-visualizer/
├── public/
│   └── gifs/
│       ├── microplastic_timeseries.gif
│       ├── microplastic_timeseries_cropped.gif
│       └── clustering_red_timeseries.gif
├── src/
│   ├── components/
│   │   ├── MicroplasticVisualizer.tsx
│   │   ├── EnhancedMicroplasticVisualizer.tsx
│   │   ├── FrameControlledGif.tsx
│   │   └── MicroplasticVisualizer.css
│   ├── App.tsx
│   ├── App.css
│   └── index.css
└── README.md
```

## Usage Guide

### Basic Navigation
1. **Select Visualization**: Click tabs to switch between different GIF types
2. **Toggle View Mode**: Use "Simple View" or "Enhanced View" buttons
3. **Control Playback**: Use play/pause and speed controls

### Enhanced Features (Enhanced View)
1. **Time Scrubbing**: Click and drag the timeline slider
2. **Frame Stepping**: Use ⏮️ and ⏭️ for single frame navigation
3. **Jump Controls**: Use ⏪ and ⏩ for 10-second jumps
4. **Real-time Info**: Monitor current time, speed, and status

### Keyboard Shortcuts (Future Enhancement)
- `Space`: Play/Pause
- `←/→`: Frame step
- `Shift + ←/→`: 10-second jump
- `+/-`: Speed adjustment

## Technical Details

### Built With
- **React 18**: Modern React with hooks
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **CSS3**: Modern styling with animations

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Performance Optimizations
- Lazy loading of GIF files
- Optimized re-renders with React hooks
- Smooth 60fps animations
- Responsive image sizing

## Data Sources

The visualizations are based on:
- **CYGNSS satellite data**: Microplastic concentration measurements
- **Time series analysis**: Temporal pattern recognition
- **Clustering algorithms**: Spatial distribution analysis

## Future Enhancements

### Planned Features
- [ ] Frame-by-frame GIF extraction for true timeline control
- [ ] Data export functionality (images, CSV)
- [ ] Zoom and pan capabilities
- [ ] Multi-visualization comparison view
- [ ] Keyboard shortcuts
- [ ] Full-screen mode
- [ ] Animation loop controls
- [ ] Custom speed presets

### Technical Improvements
- [ ] WebGL acceleration for large datasets
- [ ] Progressive loading for better performance
- [ ] Offline caching
- [ ] Advanced filtering options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact the development team.

---

**Note**: This application is designed for scientific visualization and research purposes. The GIF files contain actual satellite data and should be handled appropriately for research integrity.
