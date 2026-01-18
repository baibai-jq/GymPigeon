import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Analysis.css';

function Analysis() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [currentExercise, setCurrentExercise] = useState<string>('squat');
  const [lastEvaluation, setLastEvaluation] = useState<any>(null);
  const [reps, setReps] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const navigate = useNavigate();
  const location = useLocation();

  // Get exercise from navigation state, default to squat
  useEffect(() => {
    const state = location.state as { exercise?: string } | null;
    if (state?.exercise) {
      const exerciseMap: { [key: string]: string } = {
        'squat': 'squat',
        'bench': 'bench_press',
        'pushup': 'pushup',
        'bench_press': 'bench_press'
      };
      const normalizedExercise = exerciseMap[state.exercise] || 'squat';
      setCurrentExercise(normalizedExercise);
    }
  }, [location.state]);

  useEffect(() => {
    // Initialize WebSocket connection
    socketRef.current = new WebSocket("ws://localhost:8000/ws");

    socketRef.current.onopen = () => {
      // Send exercise selection as soon as connection opens
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        const exerciseMsg = JSON.stringify({ exercise: currentExercise });
        socketRef.current.send(exerciseMsg);
      }
    };

    socketRef.current.onmessage = (event) => {
      // Check if this is an evaluation or a frame
      if (event.data.startsWith('EVAL:')) {
        // Parse evaluation data
        try {
          const evalJson = event.data.slice(5); // Remove 'EVAL:' prefix
          const evaluation = JSON.parse(evalJson);
          setLastEvaluation(evaluation);
          if (evaluation.rep_number) {
            setReps(evaluation.rep_number);
          }
        } catch (e) {
          console.error('Failed to parse evaluation:', e);
        }
      } else {
        // It's a frame - set as image source
        setImageSrc(`data:image/jpeg;base64,${event.data}`);
        setIsReady(true);
      }
    };

    // Cleanup on component unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [currentExercise]);

  return (
    <div className="analysis-container">
      {/* Video feed */}
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

      {/* Feedback Panel */}
      {lastEvaluation && (
        <div style={{
          position: 'absolute',
          bottom: 80,
          left: 20,
          right: 20,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '15px',
          borderRadius: '8px',
          fontSize: '14px',
          zIndex: 10
        }}>
          <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>
            Rep #{lastEvaluation.rep_number} - {lastEvaluation.is_correct ? '✓ GOOD' : '✗ CHECK FORM'}
          </div>
          {lastEvaluation.errors && lastEvaluation.errors.length > 0 && (
            <div style={{ color: '#ff6b6b', marginBottom: '8px' }}>
              Issues:
              <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
                {lastEvaluation.errors.slice(0, 2).map((err: string, i: number) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}
          {lastEvaluation.warnings && lastEvaluation.warnings.length > 0 && (
            <div style={{ color: '#ffd93d' }}>
              Tips:
              <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
                {lastEvaluation.warnings.slice(0, 1).map((warn: string, i: number) => (
                  <li key={i}>{warn}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Analysis;