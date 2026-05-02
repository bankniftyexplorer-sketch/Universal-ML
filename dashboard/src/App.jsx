import { useState, useEffect, useRef } from 'react';
import { Activity, RefreshCw, Terminal, ChevronDown, CheckCircle2, AlertTriangle, Play, X } from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:4000/api';

function App() {
  const [symbols, setSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [report, setReport] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [stages, setStages] = useState({ queue: null, vault_sync: null, inference: null, training: null });
  const [failedSymbols, setFailedSymbols] = useState(new Set());
  const [isBackendDown, setIsBackendDown] = useState(false);
  const terminalRef = useRef(null);
  const activeEventSourceRef = useRef(null);

  const fetchReport = async (sym) => {
    try {
      const res = await fetch(`${API_URL}/report/${sym}`);
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setReport(data);
      setIsBackendDown(false);

      // Auto-trigger if missing or older than 12 hours (43200000 ms)
      const isStale = !data.lastUpdated || (Date.now() - new Date(data.lastUpdated).getTime()) > 43200000;
      if (isStale) {
        setFailedSymbols(prev => {
            if (!prev.has(sym)) {
                runInferenceForSymbol(sym);
            }
            return prev;
        });
      }
    } catch (err) {
      console.error(err);
      setReport(null);
      setIsBackendDown(true);
    }
  };

  const handleSymbolChange = async (sym) => {
    if (activeEventSourceRef.current) {
        activeEventSourceRef.current.close();
        activeEventSourceRef.current = null;
    }
    setSelectedSymbol(sym);
    setLogs([]);
    setIsRunning(false);
    setStages({ queue: null, vault_sync: null, inference: null, training: null });
    fetchReport(sym);
  };

  const runInferenceForSymbol = (sym) => {
    if (!sym || isRunning) return;

    setIsRunning(true);
    setStages({ queue: null, vault_sync: null, inference: null, training: null });
    setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), text: `>>> Initiating pipeline for ${sym}...` }]);

    if (activeEventSourceRef.current) {
        activeEventSourceRef.current.close();
    }

    const eventSource = new EventSource(`${API_URL}/run/${sym}`);
    activeEventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      if (parsed.type === 'log') {
        setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), text: parsed.message }]);
      } else if (parsed.type === 'stage') {
        setStages(prev => ({ ...prev, [parsed.stage]: parsed.status }));
      } else if (parsed.type === 'done') {
        setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), text: `>>> Process exited with code ${parsed.code}` }]);
        setIsRunning(false);
        eventSource.close();

        setFailedSymbols(prev => {
            const next = new Set(prev);
            if (parsed.code !== 0) next.add(sym);
            else next.delete(sym);
            return next;
        });

        // Give the file system a moment to sync, then refetch silently
        setTimeout(async () => {
          try {
            const res = await fetch(`${API_URL}/report/${sym}`);
            const data = await res.json();
            setReport(data);
          } catch(e) {}
        }, 1000);

        if (activeEventSourceRef.current === eventSource) {
            activeEventSourceRef.current = null;
        }
      }
    };

    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      eventSource.close();
      setIsRunning(false);
      setFailedSymbols(prev => new Set(prev).add(sym));
      setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), text: `>>> Connection error occurred.` }]);
      setIsBackendDown(true);
    };
  };

  // Fetch symbols on mount and retry if backend is down
  useEffect(() => {
    let currentSelected = selectedSymbol;

    const fetchSymbols = async () => {
        try {
            const res = await fetch(`${API_URL}/symbols`);
            if (!res.ok) throw new Error();
            const data = await res.json();
            setSymbols(data.symbols || []);
            setIsBackendDown(false);
            if (data.symbols && data.symbols.length > 0 && !currentSelected) {
                currentSelected = data.symbols[0];
                setSelectedSymbol(currentSelected);
                fetchReport(currentSelected);
            }
        } catch (err) {
            console.error("Error fetching symbols:", err);
            setIsBackendDown(true);
        }
    };

    fetchSymbols();
    const interval = setInterval(fetchSymbols, 5000);
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto scroll logs without moving the page window
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const getStatusColor = (confidence) => {
    if (confidence >= 0.8) return 'var(--success)';
    if (confidence >= 0.5) return 'orange';
    return 'var(--danger)';
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>

      {/* HEADER */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }} className="animate-slide-up">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))', padding: '12px', borderRadius: '12px' }}>
            <Activity size={28} color="#000" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: '700', letterSpacing: '-0.5px', margin: 0 }}>Universal-ML Command</h1>
            <p style={{ color: 'var(--text-muted)', margin: '4px 0 0 0', fontSize: '0.9rem' }}>Volatility Engine Live Dashboard</p>
          </div>
        </div>

        {/* SELECTOR */}
        <div style={{ position: 'relative', width: '250px' }}>
          <select
            value={selectedSymbol}
            onChange={(e) => handleSymbolChange(e.target.value)}
            style={{
              width: '100%',
              padding: '12px 16px',
              borderRadius: '8px',
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid var(--border-color)',
              color: 'white',
              fontSize: '1rem',
              outline: 'none',
              appearance: 'none',
              cursor: 'pointer'
            }}
          >
            <option value="" disabled>Select Symbol...</option>
            {symbols.map(s => <option key={s} value={s} style={{background: '#1a1a1a'}}>{s}</option>)}
          </select>
          <ChevronDown size={18} style={{ position: 'absolute', right: '16px', top: '14px', pointerEvents: 'none', color: 'var(--text-muted)' }} />
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>

        {/* LEFT COLUMN: METRICS & CONTROLS */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

          <div className="glass" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h2 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={20} color={report?.data ? 'var(--success)' : 'var(--text-muted)'} />
                Forecast Status
              </h2>
              {report?.lastUpdated && (
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Updated: {new Date(report.lastUpdated).toLocaleString()}
                </span>
              )}
            </div>

            {report?.data ? (
              <div className="animate-fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '4px' }}>Reference Price</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '600' }}>{report.data.reference_price.toFixed(2)}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '4px' }}>System Confidence</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '600', color: getStatusColor(report.data.forecast_confidence) }}>
                    {(report.data.forecast_confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '4px' }}>1D Peak (90%)</div>
                  <div style={{ fontSize: '1.2rem', color: 'var(--danger)' }}>{report.data.projected_peak.toFixed(2)}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '4px' }}>1D Bottom (90%)</div>
                  <div style={{ fontSize: '1.2rem', color: 'var(--success)' }}>{report.data.projected_bottom.toFixed(2)}</div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                <AlertTriangle size={32} style={{ margin: '0 auto 1rem auto', opacity: 0.5 }} />
                <p>No valid forecast found for {selectedSymbol}.</p>
              </div>
            )}

            <button
              onClick={() => runInferenceForSymbol(selectedSymbol)}
              disabled={isRunning || !selectedSymbol}
              style={{
                marginTop: '2rem',
                width: '100%',
                padding: '14px',
                borderRadius: '8px',
                border: 'none',
                background: isRunning ? 'rgba(255,255,255,0.1)' : 'linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))',
                color: isRunning ? 'var(--text-muted)' : '#000',
                fontWeight: '600',
                fontSize: '1rem',
                cursor: (isRunning || !selectedSymbol) ? 'not-allowed' : 'pointer',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.2s ease',
                boxShadow: isRunning ? 'none' : '0 4px 15px rgba(0, 242, 254, 0.3)'
              }}
            >
              {isRunning ? <RefreshCw size={18} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={18} />}
              {isRunning ? 'Updating Pipeline...' : 'Generate New Forecast'}
            </button>
            <style dangerouslySetInnerHTML={{__html: `
              @keyframes spin { 100% { transform: rotate(360deg); } }
            `}} />
          </div>

          {/* PIPELINE PROGRESS */}
          {isRunning && (
            <div className="glass animate-fade-in" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <h3 style={{ fontSize: '1rem', margin: 0, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity size={16} color="var(--accent-cyan)" /> Pipeline Status
              </h3>

              {stages.queue !== null && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {stages.queue === 'running' ? <RefreshCw size={14} className="animate-spin" color="var(--accent-blue)" /> : <CheckCircle2 size={14} color="var(--success)" />}
                  <span style={{ color: stages.queue === 'running' ? 'var(--accent-blue)' : 'var(--text-muted)' }}>Job Queued</span>
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {stages.vault_sync === 'running' ? <RefreshCw size={14} className="animate-spin" color="var(--accent-cyan)" /> :
                 stages.vault_sync === 'failed' ? <AlertTriangle size={14} color="var(--danger)" /> :
                 stages.vault_sync === 'done' ? <CheckCircle2 size={14} color="var(--success)" /> :
                 <div style={{ width: 14, height: 14, borderRadius: '50%', border: '2px solid var(--border-color)' }} />}
                <span style={{ color: stages.vault_sync ? 'white' : 'var(--text-muted)' }}>Data Vault Sync</span>
              </div>

              {stages.training !== null && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {stages.training === 'running' ? <RefreshCw size={14} className="animate-spin" color="var(--accent-cyan)" /> :
                 stages.training === 'failed' ? <AlertTriangle size={14} color="var(--danger)" /> :
                 stages.training === 'done' ? <CheckCircle2 size={14} color="var(--success)" /> :
                 <div style={{ width: 14, height: 14, borderRadius: '50%', border: '2px solid var(--border-color)' }} />}
                <span style={{ color: stages.training ? 'white' : 'var(--text-muted)' }}>Model Training</span>
              </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {stages.inference === 'running' ? <RefreshCw size={14} className="animate-spin" color="var(--accent-cyan)" /> :
                 stages.inference === 'failed' ? <AlertTriangle size={14} color="var(--danger)" /> :
                 stages.inference === 'done' ? <CheckCircle2 size={14} color="var(--success)" /> :
                 <div style={{ width: 14, height: 14, borderRadius: '50%', border: '2px solid var(--border-color)' }} />}
                <span style={{ color: stages.inference ? 'white' : 'var(--text-muted)' }}>Volatility Inference</span>
              </div>
            </div>
          )}

          {/* TERMINAL LOGS */}
          <div className="glass" style={{ height: '450px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Terminal size={18} color="var(--accent-cyan)" />
              <span style={{ fontSize: '0.9rem', fontWeight: '500' }}>Inference Engine Logs</span>
            </div>
            <div ref={terminalRef} style={{ flex: 1, padding: '1rem', overflowY: 'auto', background: '#050508', fontFamily: 'monospace', fontSize: '0.9rem', color: '#a0a0b0', whiteSpace: 'pre-wrap' }}>
              {logs.length === 0 ? (
                <div style={{ opacity: 0.5, fontStyle: 'italic' }}>System idle. Select a symbol and run inference.</div>
              ) : (
                logs.map((l, i) => (
                  <div key={i} style={{ marginBottom: '4px', lineHeight: '1.4' }}>
                    <span style={{ color: '#4facfe', marginRight: '8px' }}>[{l.time}]</span>
                    {l.text}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: CHART / IMAGE */}
        <div className="glass" style={{ padding: '1.5rem', alignSelf: 'start' }}>
          <div style={{ paddingBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '1.5rem' }}>
                <button
                  onClick={() => setActiveTab('dashboard')}
                  style={{
                      background: 'none', border: 'none', cursor: 'pointer', padding: '0 0 6px 0', margin: 0,
                      fontSize: '1.1rem', fontWeight: 'bold',
                      color: activeTab === 'dashboard' ? 'white' : 'var(--text-muted)',
                      borderBottom: activeTab === 'dashboard' ? '2px solid var(--accent-cyan)' : '2px solid transparent',
                      transition: 'all 0.2s ease'
                  }}
                >Live Forecast</button>
                <button
                  onClick={() => setActiveTab('backtest')}
                  style={{
                      background: 'none', border: 'none', cursor: 'pointer', padding: '0 0 6px 0', margin: 0,
                      fontSize: '1.1rem', fontWeight: 'bold',
                      color: activeTab === 'backtest' ? 'white' : 'var(--text-muted)',
                      borderBottom: activeTab === 'backtest' ? '2px solid var(--accent-cyan)' : '2px solid transparent',
                      transition: 'all 0.2s ease'
                  }}
                >Backtest Report</button>
            </div>
            {(activeTab === 'dashboard' ? report?.pngExists : report?.backtestExists) && (
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ fontSize: '0.9rem' }}>🔍</span> Click to zoom
              </span>
            )}
          </div>

          <div
            onClick={() => {
                if (activeTab === 'dashboard' && report?.pngExists) setIsFullscreen(true);
                if (activeTab === 'backtest' && report?.backtestExists) setIsFullscreen(true);
            }}
            style={{
              width: '100%',
              height: '340px',
              background: '#0a0a0f',
              borderRadius: '8px',
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid rgba(255,255,255,0.05)',
              cursor: (activeTab === 'dashboard' ? report?.pngExists : report?.backtestExists) ? 'zoom-in' : 'default',
              transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
              boxShadow: (activeTab === 'dashboard' ? report?.pngExists : report?.backtestExists) ? '0 0 0 0 transparent' : 'none',
            }}
            onMouseEnter={e => { if ((activeTab === 'dashboard' ? report?.pngExists : report?.backtestExists)) { e.currentTarget.style.borderColor = 'var(--accent-cyan)'; e.currentTarget.style.boxShadow = '0 0 20px rgba(0,242,254,0.15)'; } }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.05)'; e.currentTarget.style.boxShadow = 'none'; }}
          >
            {activeTab === 'dashboard' ? (
                report?.pngExists ? (
                  <img
                    src={`${API_URL}/image/${selectedSymbol}?type=dashboard&t=${report?.lastUpdated ? new Date(report.lastUpdated).getTime() : 0}`}
                    alt={`${selectedSymbol} Live Forecast`}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    className="animate-fade-in"
                  />
                ) : (
                  <div style={{ color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                    <Activity size={48} style={{ opacity: 0.2 }} />
                    <span>No forecast generated yet.</span>
                  </div>
                )
            ) : (
                report?.backtestExists ? (
                  <img
                    src={`${API_URL}/image/${selectedSymbol}?type=backtest&t=${report?.lastUpdated ? new Date(report.lastUpdated).getTime() : 0}`}
                    alt={`${selectedSymbol} Backtest Report`}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    className="animate-fade-in"
                  />
                ) : (
                  <div style={{ color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                    <Activity size={48} style={{ opacity: 0.2 }} />
                    <span>No backtest report generated yet.</span>
                  </div>
                )
            )}
          </div>
        </div>
      </div>

      {/* FULLSCREEN IMAGE MODAL */}
      {isFullscreen && (activeTab === 'dashboard' ? report?.pngExists : report?.backtestExists) && (
        <div
          onClick={() => setIsFullscreen(false)}
          style={{
            position: 'fixed',
            top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.92)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'zoom-out',
            padding: '2rem'
          }}
        >
          <img
            src={`${API_URL}/image/${selectedSymbol}?type=${activeTab}&t=${report?.lastUpdated ? new Date(report.lastUpdated).getTime() : 0}`}
            alt="Fullscreen Visual"
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            className="animate-fade-in"
          />
          <button style={{ position: 'absolute', top: '2rem', right: '2rem', background: 'transparent', border: 'none', color: 'white', cursor: 'pointer' }}>
            <X size={32} />
          </button>
        </div>
      )}

      {/* ERROR BOUNDARY MODAL */}
      {isBackendDown && (
        <div style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.95)', zIndex: 10000,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            color: 'white', backdropFilter: 'blur(10px)'
        }} className="animate-fade-in">
            <AlertTriangle size={64} color="var(--danger)" style={{ marginBottom: '1rem' }} />
            <h2 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Backend Unreachable</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>Attempting to reconnect...</p>
            <RefreshCw size={24} className="animate-spin" />
        </div>
      )}

    </div>
  );
}

export default App;
