import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Analysis.css';

interface FeedbackData {
  exercise: string;
  reps: number;
  phase: string;
  warning: string | null;
  status: string | null;
  errors: string[];
  has_evaluation: boolean;
  is_correct?: boolean;
}

function Analysis() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [currentExercise, setCurrentExercise] = useState<string>('squat');
  const [feedback, setFeedback] = useState<FeedbackData | null>(null);
  const [lastEvaluation, setLastEvaluation] = useState<any>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Initialize WebSocket connection
    socketRef.current = new WebSocket("ws://localhost:8000/ws");

    socketRef.current.onopen = () => {
      // Get exercise from navigation state
      const state = location.state as { exercise?: string } | null;
      if (state?.exercise && socketRef.current) {
        // Map exercise ID to backend format
        const exerciseMap: { [key: string]: string } = {
          'squat': 'squat',
          'pushup': 'pushup',
          'bench': 'bench_press'
        };
        const exerciseName = exerciseMap[state.exercise] || state.exercise;
        setCurrentExercise(state.exercise);
        
        // Send exercise selection to backend
        const message = JSON.stringify({ exercise: exerciseName });
        socketRef.current.send(message);
        console.log(`Sent exercise to backend: ${exerciseName}`);
      }
    };

    socketRef.current.onmessage = (event) => {
      const message = event.data;
      
      // Handle evaluation messages (start with "EVAL:")
      if (message.startsWith('EVAL:')) {
        try {
          const evalData = JSON.parse(message.substring(5));
          console.log('Evaluation received:', evalData);
          setLastEvaluation(evalData);
          
          // Clear feedback and evaluation panel after 4 seconds
          setTimeout(() => {
            setLastEvaluation(null);
            setFeedback(null);
          }, 4000);
        } catch (e) {
          console.error('Error parsing evaluation:', e);
        }
      }
      // Handle feedback messages (start with "FEEDBACK:")
      else if (message.startsWith('FEEDBACK:')) {
        try {
          const feedbackData = JSON.parse(message.substring(9));
          console.log('Feedback received:', feedbackData);
          setFeedback(feedbackData);
        } catch (e) {
          console.error('Error parsing feedback:', e);
        }
      }
      // Regular image frames (Base64 encoded)
      else {
        setImageSrc(`data:image/jpeg;base64,${message}`);
        setIsReady(true);
      }
    };

    // Cleanup on component unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [location.state]);

  const getExerciseDisplayName = () => {
    const names: { [key: string]: string } = {
      'squat': 'Barbell Squat',
      'pushup': 'Push Up',
      'bench': 'Bench Press'
    };
    return names[currentExercise] || 'Exercise';
  };

  return (
    <div className="analysis-container">
      {/* Header Controls */}
      <div className="ui-controls">
        <div className="session-header">
          <h2>Live Form Analysis</h2>
          <p>{getExerciseDisplayName()}</p>
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div className="status-badge">
            {isReady ? 'Live' : 'Connecting'}
          </div>
          <button className="exit-btn" onClick={() => navigate('/home')}>
            End Session
          </button>
        </div>
      </div>

      {/* Video Feed Container */}
      <div className="video-container">
        {imageSrc ? (
          <img 
            src={imageSrc} 
            alt="Live Stream" 
            className="video-feed" 
          />
        ) : (
          <div className="video-feed waiting-overlay">
            <div className="loading-spinner"></div>
            <p>Connecting to camera...</p>
          </div>
        )}

        {/* Feedback Overlay */}
        {feedback && (
          <div className="feedback-overlay">
            {/* Current State Stats */}
            <div className="stats-panel">
              <div className="stat-item">
                <span className="stat-label">Reps</span>
                <span className="stat-value">{feedback.reps}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Phase</span>
                <span className="stat-value">{feedback.phase}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Status</span>
                <span className={`stat-value ${feedback.status?.includes('GOOD') ? 'good' : feedback.status?.includes('CHECK') ? 'check' : ''}`}>
                  {feedback.status || '—'}
                </span>
              </div>
            </div>

            {/* Warnings */}
            {feedback.warning && (
              <div className="warning-banner">
                <span>⚠️ {feedback.warning}</span>
              </div>
            )}

            {/* Form Errors */}
            {feedback.errors.length > 0 && (
              <div className="errors-panel">
                <h4>Form Issues:</h4>
                <ul>
                  {feedback.errors.map((error, idx) => (
                    <li key={idx}>{error}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Evaluation Panel (shown after rep completed) */}
        {lastEvaluation && lastEvaluation.is_correct !== undefined && (
          <div className={`evaluation-panel ${lastEvaluation.is_correct ? 'good' : 'bad'}`}>
            <h3>Rep {lastEvaluation.rep_number} {lastEvaluation.is_correct ? '✓ GOOD' : '✗ CHECK'}</h3>
            {lastEvaluation.feedback && (
              <p className="feedback-text">{lastEvaluation.feedback}</p>
            )}
            {lastEvaluation.errors?.length > 0 && (
              <div className="errors-list">
                {lastEvaluation.errors.map((err: string, idx: number) => (
                  <div key={idx} className="error-item">• {err}</div>
                ))}
              </div>
            )}
          </div>
        )}
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