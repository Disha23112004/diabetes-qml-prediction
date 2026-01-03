import React, { useState, useEffect } from 'react';
import { Activity, Pill, AlertCircle, CheckCircle, Loader, Zap, Sparkles, TrendingUp, Brain } from 'lucide-react';

const DiabetesQMLPredictor = () => {
  const [patientData, setPatientData] = useState({
    pregnancies: '',
    glucose: '',
    bloodPressure: '',
    skinThickness: '',
    insulin: '',
    bmi: '',
    diabetesPedigreeFunction: '',
    age: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [particles, setParticles] = useState([]);

  // Generate floating particles
  useEffect(() => {
    const newParticles = Array.from({ length: 20 }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      delay: Math.random() * 5,
      duration: 3 + Math.random() * 4
    }));
    setParticles(newParticles);
  }, []);

  const predictDiabetes = async (data) => {
    setLoading(true);
    setError(null);
    setShowResults(false);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pregnancies: parseFloat(data.pregnancies),
          glucose: parseFloat(data.glucose),
          bloodPressure: parseFloat(data.bloodPressure),
          skinThickness: parseFloat(data.skinThickness),
          insulin: parseFloat(data.insulin),
          bmi: parseFloat(data.bmi),
          diabetesPedigreeFunction: parseFloat(data.diabetesPedigreeFunction),
          age: parseFloat(data.age)
        })
      });

      if (!response.ok) {
        throw new Error('Prediction failed. Is the API server running?');
      }

      const result = await response.json();
      setPrediction(result);
      setTimeout(() => setShowResults(true), 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    // Prevent negative numbers - only allow empty string or positive numbers
    if (value === '' || (parseFloat(value) >= 0 && !isNaN(value))) {
      setPatientData(prev => ({
        ...prev,
        [field]: value
      }));
    }
  };

  const handlePredict = async () => {
    await predictDiabetes(patientData);
  };

  const loadSampleData = () => {
    setPatientData({
      pregnancies: '6',
      glucose: '148',
      bloodPressure: '72',
      skinThickness: '35',
      insulin: '0',
      bmi: '33.6',
      diabetesPedigreeFunction: '0.627',
      age: '50'
    });
  };

  const isFormValid = () => {
    return Object.values(patientData).every(val => val !== '' && !isNaN(val) && parseFloat(val) >= 0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated background particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {particles.map(particle => (
          <div
            key={particle.id}
            className="absolute w-2 h-2 bg-purple-400 rounded-full opacity-20 animate-float"
            style={{
              left: `${particle.left}%`,
              animationDelay: `${particle.delay}s`,
              animationDuration: `${particle.duration}s`
            }}
          />
        ))}
      </div>

      {/* Glowing orbs */}
      <div className="absolute top-20 left-10 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" style={{ animationDelay: '1s' }} />

      <div className="relative max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="relative">
              {/* Blood Drop with Glucose Monitor Icon */}
              <svg className="w-16 h-16 animate-pulse" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <linearGradient id="dropGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style={{stopColor: '#22d3ee', stopOpacity: 1}} />
                    <stop offset="50%" style={{stopColor: '#a855f7', stopOpacity: 1}} />
                    <stop offset="100%" style={{stopColor: '#ec4899', stopOpacity: 1}} />
                  </linearGradient>
                </defs>
                {/* Blood drop shape */}
                <path 
                  d="M50,15 C50,15 70,35 70,55 C70,67 61,75 50,75 C39,75 30,67 30,55 C30,35 50,15 50,15 Z" 
                  fill="url(#dropGradient)" 
                  stroke="#22d3ee" 
                  strokeWidth="2.5"
                />
                {/* Highlight */}
                <ellipse cx="43" cy="35" rx="8" ry="12" fill="white" opacity="0.3" />
                {/* Glucose reading display */}
                <rect x="38" y="48" width="24" height="14" rx="2" fill="white" opacity="0.9" />
                <text x="50" y="58" fontSize="10" fill="#6366f1" fontWeight="bold" textAnchor="middle">
                  120
                </text>
                {/* Plus symbol for medical */}
                <circle cx="82" cy="25" r="10" fill="#fbbf24" opacity="0.9" />
                <path d="M82,20 L82,30 M77,25 L87,25" stroke="white" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <Sparkles className="w-6 h-6 text-yellow-400 absolute -top-2 -right-2 animate-spin-slow" />
            </div>
          </div>
          <h1 className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 mb-3 animate-gradient">
            Diabetes Prediction System
          </h1>
          <p className="text-xl text-purple-300 animate-fade-in-delay">
            Powered by Quantum Machine Learning
          </p>
          <div className="flex items-center justify-center gap-2 mt-4 text-cyan-300 animate-fade-in-delay-2">
            <Zap className="w-5 h-5 animate-pulse" />
            <span className="text-sm font-semibold">Intelligent Medical Diagnosis</span>
          </div>
        </div>

        {/* Main Input Card */}
        <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-xl rounded-3xl shadow-2xl border border-purple-500/30 p-8 mb-8 transform hover:scale-[1.01] transition-all duration-300 animate-slide-up">
          {/* Status Banner */}
          <div className="bg-gradient-to-r from-purple-500/20 to-cyan-500/20 rounded-2xl p-4 mb-6 border border-purple-400/30 animate-pulse-slow">
            <div className="flex items-center gap-3 text-cyan-300">
              <Activity className="w-6 h-6 animate-pulse" />
              <span className="font-semibold text-lg">Advanced AI-Powered Medical Analysis</span>
            </div>
          </div>

          {error && (
            <div className="bg-red-500/20 border-2 border-red-400 rounded-2xl p-4 mb-6 animate-shake">
              <div className="flex items-center gap-3 text-red-300">
                <AlertCircle className="w-6 h-6 animate-bounce" />
                <div>
                  <p className="font-semibold">{error}</p>
                  <p className="text-sm mt-1">Make sure the Flask API server is running on port 5000</p>
                </div>
              </div>
            </div>
          )}

          {/* Input Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {Object.entries({
              pregnancies: 'Pregnancies',
              glucose: 'Glucose (mg/dL)',
              bloodPressure: 'Blood Pressure',
              skinThickness: 'Skin Thickness',
              insulin: 'Insulin (μU/mL)',
              bmi: 'BMI',
              diabetesPedigreeFunction: 'Pedigree Function',
              age: 'Age (years)'
            }).map(([key, label], index) => (
              <div key={key} className="animate-slide-up" style={{ animationDelay: `${index * 0.05}s` }}>
                <label className="block text-sm font-bold text-purple-300 mb-2">
                  {label}
                </label>
                <input
                  type="number"
                  min="0"
                  step={key === 'bmi' || key === 'diabetesPedigreeFunction' ? '0.1' : '1'}
                  value={patientData[key]}
                  onChange={(e) => handleInputChange(key, e.target.value)}
                  onKeyDown={(e) => {
                    // Prevent minus sign and 'e' (scientific notation)
                    if (e.key === '-' || e.key === 'e' || e.key === 'E') {
                      e.preventDefault();
                    }
                  }}
                  className="w-full px-4 py-3 bg-slate-700/50 border-2 border-purple-500/30 rounded-xl text-white placeholder-slate-400 focus:border-cyan-400 focus:ring-4 focus:ring-cyan-400/20 transition-all duration-300 hover:border-purple-400/50"
                  placeholder="Enter value (≥0)"
                />
              </div>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={handlePredict}
              disabled={!isFormValid() || loading}
              className="flex-1 group relative overflow-hidden bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 text-white py-4 px-8 rounded-xl font-bold text-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/50 hover:scale-[1.02]"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300 animate-gradient" />
              <div className="relative flex items-center justify-center gap-3">
                {loading ? (
                  <>
                    <Loader className="w-6 h-6 animate-spin" />
                    <span className="animate-pulse">Analyzing Patient Data...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-6 h-6 group-hover:animate-bounce" />
                    Analyze & Predict
                  </>
                )}
              </div>
            </button>
            
            <button
              onClick={loadSampleData}
              className="px-8 py-4 bg-slate-700/50 border-2 border-purple-500/30 text-purple-300 rounded-xl font-semibold hover:bg-slate-600/50 hover:border-purple-400 transition-all duration-300 hover:scale-105"
            >
              Load Sample
            </button>
          </div>
        </div>

        {/* Results Section */}
        {prediction && (
          <div className={`space-y-8 transition-all duration-700 ${showResults ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            {/* Main Result Card */}
            <div className={`relative overflow-hidden rounded-3xl border-4 shadow-2xl transform hover:scale-[1.02] transition-all duration-300 animate-slide-up ${
              prediction.hasDiabetes 
                ? 'bg-gradient-to-br from-red-600 to-orange-600 border-red-400' 
                : 'bg-gradient-to-br from-green-600 to-emerald-600 border-green-400'
            }`}>
              {/* Animated background pattern */}
              <div className="absolute inset-0 opacity-10">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(255,255,255,0.1),transparent_50%)] animate-pulse" />
              </div>

              <div className="relative p-8">
                <div className="flex items-start gap-6">
                  {prediction.hasDiabetes ? (
                    <AlertCircle className="w-20 h-20 text-white flex-shrink-0 animate-bounce" />
                  ) : (
                    <CheckCircle className="w-20 h-20 text-white flex-shrink-0 animate-bounce" />
                  )}
                  <div className="flex-1">
                    <h2 className="text-4xl font-black text-white mb-3 animate-fade-in">
                      {prediction.hasDiabetes ? 'Diabetes Detected' : 'No Diabetes Detected'}
                    </h2>
                    <div className="flex items-center gap-3 mb-6">
                      <TrendingUp className="w-6 h-6 text-white animate-pulse" />
                      <p className="text-2xl text-white font-bold">
                        Model Confidence: {(prediction.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    
                    {/* Risk Score Progress Bar */}
                    <div className="bg-white/20 backdrop-blur-sm rounded-2xl p-6 animate-fade-in-delay">
                      <p className="text-white font-bold mb-3 flex items-center gap-2">
                        <Sparkles className="w-5 h-5 animate-spin-slow" />
                        Risk Score Analysis
                      </p>
                      <div className="relative w-full h-6 bg-white/20 rounded-full overflow-hidden">
                        <div 
                          className="absolute inset-y-0 left-0 bg-gradient-to-r from-white to-yellow-200 rounded-full shadow-lg transition-all duration-1000 ease-out animate-progress"
                          style={{ width: `${Math.min(prediction.riskScore * 100, 100)}%` }}
                        >
                          <div className="absolute inset-0 bg-white/30 animate-shimmer" />
                        </div>
                      </div>
                      <p className="text-white font-semibold mt-3 text-right">
                        {(prediction.riskScore * 100).toFixed(1)}% Risk Level
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Medicine Cards */}
            {prediction.hasDiabetes && prediction.medicines.length > 0 && (
              <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-xl rounded-3xl border border-purple-500/30 p-8 shadow-2xl animate-slide-up">
                <div className="flex items-center gap-4 mb-6">
                  <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl animate-pulse">
                    <Pill className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                    Personalized Medicine Recommendations
                  </h3>
                </div>
                
                <div className="bg-yellow-500/10 border-l-4 border-yellow-400 rounded-lg p-4 mb-6 animate-fade-in">
                  <p className="text-yellow-300 text-sm">
                    <strong>⚕️ Medical Disclaimer:</strong> AI-generated recommendations based on clinical guidelines. Always consult with a qualified healthcare professional before starting any medication.
                  </p>
                </div>

                <div className="space-y-4">
                  {prediction.medicines.map((medicine, idx) => (
                    <div 
                      key={idx} 
                      className="group bg-gradient-to-r from-purple-900/30 to-pink-900/30 border-2 border-purple-500/30 rounded-2xl p-6 hover:border-purple-400 hover:shadow-xl hover:shadow-purple-500/20 transition-all duration-300 hover:scale-[1.02] animate-slide-up"
                      style={{ animationDelay: `${idx * 0.1}s` }}
                    >
                      <h4 className="text-2xl font-bold text-purple-300 mb-4 group-hover:text-purple-200 transition-colors">
                        💊 {medicine.name}
                      </h4>
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="bg-slate-800/50 rounded-xl p-4">
                          <p className="text-sm text-purple-400 font-semibold mb-1">Dosage</p>
                          <p className="text-white text-lg font-bold">{medicine.dosage}</p>
                        </div>
                        <div className="bg-slate-800/50 rounded-xl p-4">
                          <p className="text-sm text-purple-400 font-semibold mb-1">Frequency</p>
                          <p className="text-white text-lg font-bold">{medicine.frequency}</p>
                        </div>
                      </div>
                      <div className="bg-purple-500/10 rounded-xl p-4 mb-3">
                        <p className="text-purple-200">{medicine.notes}</p>
                      </div>
                      {medicine.mechanism && (
                        <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 rounded-xl p-4 border border-cyan-500/20">
                          <p className="text-xs text-cyan-400 font-bold mb-1">⚗️ MECHANISM OF ACTION</p>
                          <p className="text-cyan-200 text-sm">{medicine.mechanism}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Lifestyle Recommendations */}
            <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-xl rounded-3xl border border-purple-500/30 p-8 shadow-2xl animate-slide-up">
              <h3 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-400 mb-6">
                🌟 Lifestyle Recommendations
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {prediction.lifestyle.map((item, idx) => (
                  <div 
                    key={idx} 
                    className="flex items-start gap-3 bg-gradient-to-r from-green-900/20 to-emerald-900/20 p-4 rounded-xl border border-green-500/20 hover:border-green-400/50 transition-all duration-300 hover:scale-105 animate-fade-in"
                    style={{ animationDelay: `${idx * 0.05}s` }}
                  >
                    <CheckCircle className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5 animate-pulse" />
                    <span className="text-slate-200 leading-relaxed">{item}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) translateX(0); }
          50% { transform: translateY(-20px) translateX(10px); }
        }
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @keyframes progress {
          from { width: 0; }
        }
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-10px); }
          75% { transform: translateX(10px); }
        }
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slide-up {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-float { animation: float linear infinite; }
        .animate-gradient { 
          background-size: 200% 200%;
          animation: gradient 3s ease infinite;
        }
        .animate-shimmer { animation: shimmer 2s infinite; }
        .animate-progress { animation: progress 1s ease-out; }
        .animate-shake { animation: shake 0.5s; }
        .animate-fade-in { animation: fade-in 0.6s ease-out; }
        .animate-fade-in-delay { animation: fade-in 0.6s ease-out 0.2s both; }
        .animate-fade-in-delay-2 { animation: fade-in 0.6s ease-out 0.4s both; }
        .animate-slide-up { animation: slide-up 0.6s ease-out both; }
        .animate-pulse-slow { animation: pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        .animate-spin-slow { animation: spin 3s linear infinite; }
      `}</style>
    </div>
  );
};

export default DiabetesQMLPredictor;