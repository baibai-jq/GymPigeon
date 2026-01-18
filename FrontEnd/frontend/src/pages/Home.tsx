import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';
const BarbellSquat = 'BarbellSquat.png'

const SAVED_WORKOUTS = [
  { id: 1, title: "Morning Squat Blitz", duration: "15 min", difficulty: "Easy", reps: 50 },
  { id: 2, title: "Bicep Burner", duration: "10 min", difficulty: "Medium", reps: 30 },
  { id: 3, title: "Full Body AI Check", duration: "25 min", difficulty: "Hard", reps: 100 },
];

const RECENT_ACTIVITY = [
  { date: "Yesterday", exercise: "Squats", score: "95% Form" },
  { date: "2 days ago", exercise: "Bicep Curls", score: "88% Form" },
];

function Home() {
  const navigate = useNavigate();
  const [isPopupOpen, setIsPopupOpen] = useState(false);

  const exercises = [
    { id: 'squat', name: 'Barbell Squat', image: BarbellSquat },
    { id: 'deadlift', name: 'Deadlift', image: BarbellSquat },
    { id: 'bench', name: 'Bench Press', image: BarbellSquat }
  ];

  const handleExerciseSelect = (id: string) => {
    setIsPopupOpen(false);
    navigate('/analysis', { state: { exercise: id } });
  };

  return (
    <div className="home-container">
      {/* Header */}
      <header className="home-header">
        <div className="header-text">
          <h1>Welcome back!</h1>
          <p>Your gym buddy is ready for today's session.</p>
        </div>
        <div className="points-badge">450 pts</div>
      </header>

      {/* Stats Dashboard */}
      <section className="stats-grid">
        <div className="stat-card">
          <h3>Total Reps</h3>
          <p className="stat-number">1,240</p>
        </div>
        <div className="stat-card">
          <h3>Avg. Form</h3>
          <p className="stat-number">92%</p>
        </div>
        <div className="stat-card">
          <h3>Streak</h3>
          <p className="stat-number">5🔥</p>
        </div>
      </section>

      {/* Saved Workouts */}
      <section className="section-container">
        <h2>Saved Workout Programs</h2>
        <div className="workout-grid">
          {SAVED_WORKOUTS.map((workout) => (
            <div key={workout.id} className="workout-card">
              <div className="workout-info">
                <h4>{workout.title}</h4>
                <span>{workout.duration} • {workout.difficulty}</span>
              </div>
              <button className="card-start-btn" onClick={() => setIsPopupOpen(true)}>Start</button>
            </div>
          ))}
        </div>
      </section>

      {/* Recent History */}
      <section className="section-container" style={{ paddingBottom: '120px' }}>
        <h2>Recent Activity</h2>
        <div className="history-list">
          {RECENT_ACTIVITY.map((act, i) => (
            <div key={i} className="history-item">
              <span>{act.date}</span>
              <strong>{act.exercise}</strong>
              <span className="form-score">{act.score}</span>
            </div>
          ))}
        </div>
      </section>

      {/* FLOATING ACTION BAR */}
      <div className="floating-nav-bar">
        <button className="main-action-btn" onClick={() => setIsPopupOpen(true)}>
          <span className="plus-icon">+</span> Start AI Analysis
        </button>
      </div>

      {/* POPUP MODAL */}
      {isPopupOpen && (
        <div className="modal-overlay" onClick={() => setIsPopupOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Select Exercise</h3>
            <div className="exercise-grid">
              {exercises.map((ex) => (
                <div key={ex.id} className="exercise-card" onClick={() => handleExerciseSelect(ex.id)}>
                  <div className="image-container">
                    <img src={ex.image} alt={ex.name} className='exercise-image' />
                  </div>
                  <div className='card-footer'>
                    <span className="name">{ex.name}</span>
                  </div>
                </div>
              ))}
            </div>
            <button className="close-btn" onClick={() => setIsPopupOpen(false)}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Home;