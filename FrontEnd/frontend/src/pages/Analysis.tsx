import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Analysis.css';

function Analysis() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [currentExercise, setCurrentExercise] = useState<string>('squat');
  const [lastEvaluation, setLastEvaluation] = useState<any>(null);
  const [reps, setReps] = useState(0);
  const [showSummary, setShowSummary] = useState(false);
  const [sessionSummary, setSessionSummary] = useState<any>(null);
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
      // Check if this is an evaluation, summary, or a frame
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
      } else if (event.data.startsWith('SUMMARY:')) {
        // Parse session summary
        try {
          const summaryJson = event.data.slice(8); // Remove 'SUMMARY:' prefix
          const summary = JSON.parse(summaryJson);
          setSessionSummary(summary);
          setShowSummary(true);
        } catch (e) {
          console.error('Failed to parse summary:', e);
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

      {/* Session Summary Panel */}
      {showSummary && sessionSummary && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.9)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: '#1a1a1a',
            color: 'white',
            padding: '40px',
            borderRadius: '12px',
            maxWidth: '600px',
            maxHeight: '80vh',
            overflowY: 'auto',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.8)'
          }}>
            <h2 style={{ marginBottom: '10px', textAlign: 'center' }}>Workout Summary</h2>
            <p style={{ textAlign: 'center', color: '#aaa', marginBottom: '25px' }}>
              Exercise: {sessionSummary.exercise?.replace('_', ' ').toUpperCase()}
            </p>
            
            <div style={{
              marginBottom: '25px',
              padding: '15px',
              backgroundColor: '#2a2a2a',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#4CAF50' }}>
                {sessionSummary.total_reps}
              </div>
              <div style={{ color: '#aaa' }}>Total Reps Completed</div>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ marginBottom: '15px', fontSize: '16px' }}>Rep Breakdown</h3>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
                gap: '10px'
              }}>
                {sessionSummary.reps && sessionSummary.reps.map((rep: any, idx: number) => (
                  <div
                    key={idx}
                    style={{
                      padding: '12px',
                      borderRadius: '8px',
                      textAlign: 'center',
                      backgroundColor: rep.is_correct ? 'rgba(76, 175, 80, 0.2)' : 'rgba(244, 67, 54, 0.2)',
                      border: `2px solid ${rep.is_correct ? '#4CAF50' : '#F44336'}`,
                      cursor: 'default'
                    }}
                    title={rep.main_error || 'Good form'}
                  >
                    <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '5px' }}>
                      Rep {rep.rep_number}
                    </div>
                    <div style={{
                      fontSize: '12px',
                      color: rep.is_correct ? '#4CAF50' : '#F44336',
                      fontWeight: 'bold'
                    }}>
                      {rep.is_correct ? '✓ OK' : '⚠ Issue'}
                    </div>
                    {rep.main_error && (
                      <div style={{
                        fontSize: '10px',
                        color: '#aaa',
                        marginTop: '5px',
                        maxHeight: '40px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {rep.main_error}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={() => {
                setShowSummary(false);
                navigate('/home');
              }}
              style={{
                width: '100%',
                padding: '12px',
                marginTop: '20px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '16px',
                fontWeight: 'bold',
                cursor: 'pointer',
                transition: 'background-color 0.3s'
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#45a049')}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#4CAF50')}
            >
              Back to Home
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Analysis;