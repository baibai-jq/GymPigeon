import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

function Login() {
  const navigate = useNavigate();

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="logo-icon">🏋️‍♂️</div>
        <h1>Gym AI Coach</h1>
        <p>Master your form with real-time AI feedback.</p>
        <button className="btn-login" onClick={() => navigate('/home')}>
          Get Started
        </button>
      </div>
    </div>
  );
}

export default Login;