import React, { useState, useRef, useEffect } from 'react';
import {
  Send, Scale, Sparkles, MessageSquare, Plus,
  History, Trash2, User, Mail, Lock, Eye,
  EyeOff, AlertCircle, Loader,
} from 'lucide-react';
import { Client } from "@gradio/client";

// Import your Firebase functions
import {
  auth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  saveUserData,
  getUserData,
  saveChatLog,
  getChatHistory,
} from './firebase';

// ✅ UPDATED: Uses your actual Gradio Space and correct endpoint
const getAIResponse = async (userInput) => {
  const client = await Client.connect("Deepti-singh-196/LawAssit_Version1_RAG");

  const result = await client.predict("/respond", {
    message: userInput,
  });

  return result.data[0];
};

export default function LawAssistChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hello! I'm LawAssist, your AI-powered legal companion. How can I help you with your legal questions today?",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [modelStatus, setModelStatus] = useState('ready');
  const [sessionId, setSessionId] = useState(null);
  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState('login');
  const [showAuth, setShowAuth] = useState(true);
  const [chatHistory, setChatHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [saveStatus, setSaveStatus] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auth form states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [authError, setAuthError] = useState('');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Listen to Firebase auth state changes
  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (firebaseUser) => {
      if (firebaseUser) {
        const userDataResult = await getUserData(firebaseUser.uid);
        const userData = {
          uid: firebaseUser.uid,
          email: firebaseUser.email,
          name: userDataResult.data?.name || firebaseUser.email.split('@')[0],
        };
        setUser(userData);
        setShowAuth(false);
        await loadChatHistory(firebaseUser.uid);
      } else {
        setUser(null);
        setShowAuth(true);
      }
    });

    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (user && !sessionId) {
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setSessionId(newSessionId);
    }
  }, [user, sessionId]);

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const handleLogin = async () => {
    setAuthError('');
    setIsLoading(true);

    if (!email || !validateEmail(email)) {
      setAuthError('Please enter a valid email address');
      setIsLoading(false);
      return;
    }
    if (!password) {
      setAuthError('Please enter your password');
      setIsLoading(false);
      return;
    }

    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (error) {
      if (error.code === 'auth/user-not-found') {
        setAuthError('No account found with this email. Please sign up.');
      } else if (error.code === 'auth/wrong-password') {
        setAuthError('Incorrect password. Please try again.');
      } else if (error.code === 'auth/invalid-credential') {
        setAuthError('Invalid credentials. Please check your email and password.');
      } else if (error.code === 'auth/too-many-requests') {
        setAuthError('Too many failed attempts. Please try again later.');
      } else {
        setAuthError(error.message || 'Login failed. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignup = async () => {
    setAuthError('');
    setIsLoading(true);

    if (!name.trim()) {
      setAuthError('Please enter your name');
      setIsLoading(false);
      return;
    }
    if (!email || !validateEmail(email)) {
      setAuthError('Please enter a valid email address');
      setIsLoading(false);
      return;
    }
    if (!password || password.length < 6) {
      setAuthError('Password must be at least 6 characters');
      setIsLoading(false);
      return;
    }
    if (password !== confirmPassword) {
      setAuthError('Passwords do not match');
      setIsLoading(false);
      return;
    }

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      await saveUserData(userCredential.user.uid, {
        name: name,
        email: email,
      });
    } catch (error) {
      if (error.code === 'auth/email-already-in-use') {
        setAuthError('An account with this email already exists. Please login.');
      } else if (error.code === 'auth/weak-password') {
        setAuthError('Password is too weak. Please use a stronger password.');
      } else if (error.code === 'auth/invalid-email') {
        setAuthError('Invalid email address.');
      } else {
        setAuthError(error.message || 'Signup failed. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const saveMessage = async (message) => {
    if (!user || !sessionId) return;

    setSaveStatus('Saving...');

    try {
      const result = await saveChatLog(user.uid, sessionId, user.name, message);
      setSaveStatus(result.success ? 'Saved ✓' : 'Save failed');
    } catch (error) {
      setSaveStatus('Save failed');
    }
    setTimeout(() => setSaveStatus(''), 2000);
  };

  const loadChatHistory = async (userId) => {
    try {
      const result = await getChatHistory(userId);

      if (result.success) {
        const sessions = {};
        result.logs.forEach(log => {
          if (!sessions[log.sessionId]) sessions[log.sessionId] = [];
          sessions[log.sessionId].push(log);
        });
        const recentSessions = Object.entries(sessions)
          .sort((a, b) => new Date(b[1][0].timestamp) - new Date(a[1][0].timestamp))
          .slice(0, 5);
        setChatHistory(recentSessions);
      }
    } catch (error) {
      console.error('Exception loading chat history:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;

    const userMessage = {
      role: "user",
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    await saveMessage(userMessage);

    const userQuestion = input.trim();
    setInput("");
    setIsTyping(true);
    setModelStatus("Connecting to Hugging Face...");

    try {
      setModelStatus("Thinking...");

      const aiResponse = await getAIResponse(userQuestion);

      const aiMessage = {
        role: "assistant",
        content: typeof aiResponse === "string" ? aiResponse.trim() : JSON.stringify(aiResponse),
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, aiMessage]);
      await saveMessage(aiMessage);
      setModelStatus("ready");

    } catch (error) {
      console.error("Gradio API error:", error);
      setModelStatus("offline");

      const errorMsg = {
        role: "assistant",
        content: `⚠️ Unable to connect to the AI model.\n\nError: ${error.message}\n\nPlease check:\n1. Your Hugging Face Space "Deepti-singh-196/LawAssit_Version1_RAG" is running\n2. The Space is set to public\n3. The /respond endpoint is active`,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const startNewChat = () => {
    setMessages([{
      role: 'assistant',
      content: "Hello! I'm LawAssist, your AI-powered legal companion. How can I help you with your legal questions today?",
      timestamp: new Date().toISOString(),
    }]);
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
  };

  const clearHistory = () => {
    if (window.confirm('Clear visible history?')) {
      setChatHistory([]);
    }
  };

  const handleLogout = async () => {
    await signOut(auth);
    setMessages([{
      role: 'assistant',
      content: "Hello! I'm LawAssist, your AI-powered legal companion. How can I help you with your legal questions today?",
      timestamp: new Date().toISOString(),
    }]);
    setSessionId(null);
    setChatHistory([]);
  };

  const suggestedPrompts = [
    'What are the fundamental rights under the Indian Constitution?',
    'Explain contract law basics in India',
    'What is intellectual property law?',
    'How does copyright work in India?',
  ];

  if (showAuth) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4">
        <div className="bg-slate-800/50 backdrop-blur-lg border border-blue-500/30 rounded-3xl p-8 max-w-md w-full shadow-2xl shadow-blue-500/20">
          <div className="flex justify-center mb-6">
            <div className="bg-gradient-to-br from-blue-400 to-cyan-400 p-4 rounded-2xl shadow-lg shadow-blue-500/50">
              <Scale className="w-12 h-12 text-slate-900" />
            </div>
          </div>
          <h2 className="text-3xl font-bold text-center bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">
            {authMode === 'login' ? 'Welcome Back' : 'Create Account'}
          </h2>
          <p className="text-slate-400 text-center mb-6">
            {authMode === 'login' ? 'Login to continue with LawAssist AI' : 'Sign up to get started'}
          </p>
          {authError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 mb-4 flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-red-400 text-sm">{authError}</p>
            </div>
          )}
          <div className="space-y-4">
            {authMode === 'signup' && (
              <div>
                <label className="text-slate-300 text-sm mb-1 block">Full Name</label>
                <div className="relative">
                  <User className="absolute left-3 top-3 w-5 h-5 text-slate-500" />
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="John Doe"
                    className="w-full bg-slate-900/50 border border-slate-700 focus:border-blue-500 rounded-xl pl-10 pr-4 py-3 text-white placeholder-slate-500 focus:outline-none"
                  />
                </div>
              </div>
            )}
            <div>
              <label className="text-slate-300 text-sm mb-1 block">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 w-5 h-5 text-slate-500" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full bg-slate-900/50 border border-slate-700 focus:border-blue-500 rounded-xl pl-10 pr-4 py-3 text-white placeholder-slate-500 focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="text-slate-300 text-sm mb-1 block">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 w-5 h-5 text-slate-500" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-slate-900/50 border border-slate-700 focus:border-blue-500 rounded-xl pl-10 pr-12 py-3 text-white placeholder-slate-500 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-3 text-slate-500 hover:text-slate-300"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>
            {authMode === 'signup' && (
              <div>
                <label className="text-slate-300 text-sm mb-1 block">Confirm Password</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 w-5 h-5 text-slate-500" />
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full bg-slate-900/50 border border-slate-700 focus:border-blue-500 rounded-xl pl-10 pr-12 py-3 text-white placeholder-slate-500 focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-3 text-slate-500 hover:text-slate-300"
                  >
                    {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>
            )}
          </div>
          <button
            onClick={authMode === 'login' ? handleLogin : handleSignup}
            disabled={isLoading}
            className="w-full bg-gradient-to-br from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:from-slate-600 disabled:to-slate-600 text-white py-3 rounded-xl font-medium transition-all duration-200 mt-6 shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                {authMode === 'login' ? 'Logging in...' : 'Signing up...'}
              </>
            ) : authMode === 'login' ? 'Login' : 'Sign Up'}
          </button>
          <div className="text-center mt-6">
            <button
              onClick={() => {
                setAuthMode(authMode === 'login' ? 'signup' : 'login');
                setAuthError('');
                setEmail('');
                setPassword('');
                setConfirmPassword('');
                setName('');
              }}
              className="text-slate-400 hover:text-blue-400 text-sm transition-colors"
            >
              {authMode === 'login' ? "Don't have an account? Sign up" : 'Already have an account? Login'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      <header className="bg-slate-900/50 backdrop-blur-lg border-b border-blue-500/20 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-400 to-cyan-400 p-2 rounded-xl shadow-lg shadow-blue-500/50">
              <Scale className="w-6 h-6 text-slate-900" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                LawAssist AI
              </h1>
              <p className="text-xs text-slate-400 flex items-center gap-1">
                <User className="w-3 h-3" />
                {user?.name}
                {saveStatus && <span className="ml-2 text-green-400">{saveStatus}</span>}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={startNewChat} className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-300 hover:text-white" title="New Chat">
              <Plus className="w-5 h-5" />
            </button>
            <button onClick={() => setShowHistory(!showHistory)} className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-300 hover:text-white" title="Chat History">
              <History className="w-5 h-5" />
            </button>
            <button onClick={clearHistory} className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-red-400 hover:text-red-300" title="Clear History">
              <Trash2 className="w-5 h-5" />
            </button>
            <button onClick={handleLogout} className="px-3 py-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-300 hover:text-white text-sm" title="Logout">
              Logout
            </button>
          </div>
        </div>
      </header>

      {showHistory && (
        <div className="fixed right-4 top-20 w-80 bg-slate-800/95 backdrop-blur-lg border border-blue-500/30 rounded-2xl p-4 shadow-2xl shadow-blue-500/20 max-h-96 overflow-y-auto z-50">
          <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
            <History className="w-5 h-5 text-blue-400" />
            Recent Sessions
          </h3>
          {chatHistory.length === 0 ? (
            <p className="text-slate-400 text-sm">No chat history yet</p>
          ) : (
            <div className="space-y-2">
              {chatHistory.map(([sid, logs], idx) => (
                <div key={idx} className="bg-slate-900/50 rounded-lg p-3 hover:bg-slate-900 transition-colors cursor-pointer">
                  <p className="text-xs text-slate-500 mb-1">{new Date(logs[0].timestamp).toLocaleDateString()}</p>
                  <p className="text-sm text-slate-300">{logs.length} messages</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 1 && (
            <div className="text-center py-12 space-y-6">
              <div className="inline-block bg-gradient-to-br from-blue-500 to-cyan-500 p-4 rounded-2xl shadow-2xl shadow-blue-500/30">
                <Sparkles className="w-12 h-12 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-white">How can I assist you today?</h2>
              <p className="text-slate-400 max-w-2xl mx-auto">
                Ask me anything about Indian law, legal concepts, contracts, rights, or general legal information.
                <br />
                <span className="text-sm text-blue-400">✓ Powered by Hugging Face • Saved to Firebase</span>
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto pt-4">
                {suggestedPrompts.map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInput(prompt)}
                    className="p-4 bg-slate-800/50 hover:bg-slate-800 border border-blue-500/20 hover:border-blue-500/40 rounded-xl text-left text-slate-300 hover:text-white transition-all duration-200 group"
                  >
                    <MessageSquare className="w-4 h-4 inline mr-2 text-blue-400 group-hover:scale-110 transition-transform" />
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-3xl rounded-2xl px-6 py-4 ${
                message.role === 'user'
                  ? 'bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-lg shadow-blue-500/30'
                  : 'bg-slate-800/50 backdrop-blur-sm text-slate-100 border border-slate-700/50'
              }`}>
                {message.role === 'assistant' && (
                  <div className="flex items-center gap-2 mb-2 text-blue-400 text-sm font-medium">
                    <Scale className="w-4 h-4" />
                    LawAssist AI
                  </div>
                )}
                <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                <p className="text-xs opacity-50 mt-2">{new Date(message.timestamp).toLocaleTimeString()}</p>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl px-6 py-4">
                <div className="flex items-center gap-2 text-blue-400 text-sm font-medium mb-2">
                  <Scale className="w-4 h-4" />
                  LawAssist AI
                </div>
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="border-t border-blue-500/20 bg-slate-900/50 backdrop-blur-lg px-4 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3 items-end">
            <div className="flex-1 bg-slate-800/50 border border-slate-700 focus-within:border-blue-500 rounded-2xl transition-colors duration-200 shadow-lg">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a legal question..."
                rows="1"
                className="w-full bg-transparent text-white px-6 py-4 resize-none focus:outline-none placeholder-slate-500"
                style={{ minHeight: '56px', maxHeight: '200px' }}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isTyping}
              className="bg-gradient-to-br from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:from-slate-700 disabled:to-slate-700 text-white p-4 rounded-2xl transition-all duration-200 shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 disabled:shadow-none"
            >
              {isTyping
                ? <Loader className="w-5 h-5 animate-spin" />
                : <Send className="w-5 h-5" />}
            </button>
          </div>
          <div className="flex items-center justify-between mt-3 text-xs text-slate-500">
            <p>
              Status: <span className={
                modelStatus === 'ready' ? 'text-green-400' :
                modelStatus.includes('Thinking') || modelStatus.includes('Connecting') ? 'text-yellow-400' :
                'text-red-400'
              }>{modelStatus}</span>
            </p>
            <p className="text-slate-600">Press Enter to send, Shift+Enter for new line</p>
          </div>
        </div>
      </div>
    </div>
  );
}
