import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

function Home() {
  const navigate = useNavigate();
  // 1. Create a "State" to track if the popup is open
  const [isPopupOpen, setIsPopupOpen] = useState(false);

  const exercises = [
    { id: 'squat', name: 'Barbell Squat', icon: '🦵' },
    { id: 'deadlift', name: 'Deadlift', icon: '🏋️' },
    { id: 'bench', name: 'Bench Press', icon: '💪' }
  ];

  const handleExerciseSelect = (id: string) => {
    console.log("Selected:", id);
    setIsPopupOpen(false);
    navigate('/analysis', { state: { exercise: id } });
  };

  return (
    <div className="home-container">
      <h1>Your Workouts</h1>
      
      <div className="stats-row">
        <div className="stat-box">Points: 450</div>
        <div className="stat-box">Streak: 5🔥</div>
      </div>

      {/* 2. This button opens the popup */}
      <button className="start-btn" onClick={() => setIsPopupOpen(true)}>
        Start Analysis
      </button>

      {/* 3. THE POPUP LOGIC */}
      {isPopupOpen && (
        <div className="modal-overlay" onClick={() => setIsPopupOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Select Exercise</h3>
            <div className="exercise-grid">
              {exercises.map((ex) => (
                <div 
                  key={ex.id} 
                  className="exercise-card" 
                  onClick={() => handleExerciseSelect(ex.id)}
                >
                  <span className="icon">{ex.icon}</span>
                  <span className="name">{ex.name}</span>
                </div>
              ))}
            </div>
            <button className="close-btn" onClick={() => setIsPopupOpen(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Home;