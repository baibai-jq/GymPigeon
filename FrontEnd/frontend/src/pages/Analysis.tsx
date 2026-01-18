import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Analysis.css';

function Analysis() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    // Initialize WebSocket connection
    socketRef.current = new WebSocket("ws://localhost:8000/ws");

    socketRef.current.onmessage = (event) => {
      // Receive the Base64 string and set it as the image source
      setImageSrc(`data:image/jpeg;base64,${event.data}`);
      
      // Update status to Ready once the first frame arrives
      setIsReady(true);
    };

    // Cleanup on component unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  return (
    <div className="analysis-container">
      {/* Replaced the <video> and <canvas> stack with a single <img> 
        We use the 'video-feed' class to maintain the same size/layout as the original video.
      */}
      {imageSrc ? (
        <img 
          src={imageSrc} 
          alt="Live Stream" 
          className="video-feed" 
        />
      ) : (
        // Optional: Placeholder while waiting for connection
        <div className="video-feed" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'black', color: 'white' }}>
          <p>Waiting for server...</p>
        </div>
      )}
      
      {/* The UI Layer */}
      <div className="ui-controls">
        <button className="exit-btn" onClick={() => navigate('/home')}>
          End Session
        </button>
      </div>
    </div>
  );
}

export default Analysis;