import React, { useEffect, useRef, useState } from 'react';
import { Pose } from '@mediapipe/pose';
import type { Results } from '@mediapipe/pose';
import * as cam from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import { useNavigate } from 'react-router-dom';
import './Analysis.css';

function Analysis() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const navigate = useNavigate();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1, // Set to 0 for faster performance on mobile
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results: any) => {
      if (!canvasRef.current || !videoRef.current) return;
      
      const canvasCtx = canvasRef.current.getContext('2d');
      if (!canvasCtx) return;

      // 1. Clear the canvas (makes it transparent)
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // 2. Draw the skeleton only if landmarks are found
      if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
          { color: '#00FF00', lineWidth: 4 });
        drawLandmarks(canvasCtx, results.poseLandmarks,
          { color: '#FF0000', lineWidth: 2 });
      }
      canvasCtx.restore();
      if (!isReady) setIsReady(true);
    });

    if (videoRef.current) {
      const camera = new cam.Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current) {
            await pose.send({ image: videoRef.current });
          }
        },
        width: 1280,
        height: 720,
      });
      camera.start();
    }
  }, []);

  return (
    <div className="analysis-container">
      {/* The Camera Layer */}
      <video ref={videoRef} className="video-feed" playsInline />
      
      {/* The AI Layer (Stacked on top) */}
      <canvas ref={canvasRef} className="skeleton-canvas" width="1280" height="720" />
      
      {/* The UI Layer */}
      <div className="ui-controls">
        <div className="status-badge">
            {isReady ? "● AI ACTIVE" : "LOADING AI..."}
        </div>
        <button className="exit-btn" onClick={() => navigate('/home')}>
          End Session
        </button>
      </div>
    </div>
  );
}

export default Analysis;