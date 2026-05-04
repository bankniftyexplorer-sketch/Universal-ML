import React, { useState, useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { Activity, RefreshCw, Terminal, ChevronDown, CheckCircle2, AlertTriangle, Play, X, Zap, Shield, BarChart2 } from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:4000/api';

// ============================================================================
// HIGH-FREQUENCY TERMINAL (Decoupled to prevent React render thrashing)
// ============================================================================
const TerminalWindow = forwardRef((props, ref) => {
    const [logs, setLogs] = useState([]);
    const scrollRef = useRef(null);

    useImperativeHandle(ref, () => ({
        addLog: (text) => {
            setLogs(prev => [...prev, { time: new Date().toLocaleTimeString('en-US', { hour12: false }), text }]);
        },
        clearLogs: () => {
            setLogs([]);
        }
    }));

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="glass-panel terminal-container animate-slide-up" style={{ animationDelay: '0.2s' }}>
            <div className="terminal-header">
                <Terminal size={16} color="var(--accent-cyan)" />
                <span>Engine Telemetry</span>
                <div style={{ marginLeft: 'auto', display: 'flex', gap: '6px' }}>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#ff5f56' }}></div>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#ffbd2e' }}></div>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#27c93f' }}></div>
                </div>
            </div>
            <div className="terminal-body" ref={scrollRef}>
                {logs.length === 0 ? (
                    <div style={{ opacity: 0.3, fontStyle: 'italic', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '1rem' }}>
                        <Terminal size={48} />
                        Awaiting pipeline execution...
                    </div>
                ) : (
                    logs.map((l, i) => {
                        const isError = l.text.includes('Error') || l.text.includes('[!]');
                        const isSuccess = l.text.includes('[✓]');
                        const isHighlight = l.text.includes('===');

                        return (
                            <div key={i} className="log-line animate-fade-in">
                                <span className="log-bracket">[</span>
                                <span className="log-time">{l.time}</span>
                                <span className="log-bracket">]</span>{' '}
                                <span style={{
                                    color: isError ? 'var(--status-danger)' :
                                           isSuccess ? 'var(--status-success)' :
                                           isHighlight ? 'var(--accent-cyan)' : 'inherit',
                                    fontWeight: isHighlight ? '600' : 'normal'
                                }}>
                                    {l.text}
                                </span>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
});

// ============================================================================
// MAIN DASHBOARD APP
// ============================================================================
function App() {
    const [symbols, setSymbols] = useState([]);
    const [selectedSymbol, setSelectedSymbol] = useState('');
    const [report, setReport] = useState(null);
    const [isRunning, setIsRunning] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [activeTab, setActiveTab] = useState('dashboard');
    const [stages, setStages] = useState({ queue: null, vault_sync: null, inference: null, training: null });
    const [isBackendDown, setIsBackendDown] = useState(false);

    const terminalRef = useRef(null);
    const activeEventSourceRef = useRef(null);

    // Initial load
    useEffect(() => {
        let isMounted = true;
        const fetchSymbols = async () => {
            try {
                const res = await fetch(`${API_URL}/symbols`);
                if (!res.ok) throw new Error();
                const data = await res.json();
                if (!isMounted) return;

                setSymbols(data.symbols || []);
                setIsBackendDown(false);

                if (data.symbols && data.symbols.length > 0 && !selectedSymbol) {
                    const initialSymbol = data.symbols[0];
                    setSelectedSymbol(initialSymbol);
                    fetchReport(initialSymbol);
                }
            } catch (err) {
                if (isMounted) setIsBackendDown(true);
            }
        };

        fetchSymbols();
        return () => { isMounted = false; };
    }, []);

    const fetchReport = async (sym) => {
        try {
            const res = await fetch(`${API_URL}/report/${sym}`);
            if (!res.ok) throw new Error('Backend error');
            const data = await res.json();
            setReport(data);
            setIsBackendDown(false);
        } catch (err) {
            setReport(null);
            setIsBackendDown(true);
        }
    };

    const handleSymbolChange = (sym) => {
        if (activeEventSourceRef.current) {
            activeEventSourceRef.current.close();
            activeEventSourceRef.current = null;
        }
        setSelectedSymbol(sym);
        setIsRunning(false);
        setStages({ queue: null, vault_sync: null, inference: null, training: null });
        if (terminalRef.current) terminalRef.current.clearLogs();
        fetchReport(sym);
    };

    const runInferenceForSymbol = (sym) => {
        if (!sym || isRunning) return;

        setIsRunning(true);
        setStages({ queue: null, vault_sync: null, inference: null, training: null });
        if (terminalRef.current) {
            terminalRef.current.clearLogs();
            terminalRef.current.addLog(`>>> Initiating pipeline for ${sym}...`);
        }

        if (activeEventSourceRef.current) {
            activeEventSourceRef.current.close();
        }

        const eventSource = new EventSource(`${API_URL}/run/${sym}`);
        activeEventSourceRef.current = eventSource;

        eventSource.onmessage = (event) => {
            const parsed = JSON.parse(event.data);
            if (parsed.type === 'log') {
                if (terminalRef.current) terminalRef.current.addLog(parsed.message);
            } else if (parsed.type === 'stage') {
                setStages(prev => ({ ...prev, [parsed.stage]: parsed.status }));
            } else if (parsed.type === 'done') {
                if (terminalRef.current) terminalRef.current.addLog(`>>> Process exited with code ${parsed.code}`);
                setIsRunning(false);
                eventSource.close();
                activeEventSourceRef.current = null;

                setTimeout(() => fetchReport(sym), 1000);
            }
        };

        eventSource.onerror = (err) => {
            eventSource.close();
            setIsRunning(false);
            if (terminalRef.current) terminalRef.current.addLog(`>>> Connection dropped. Pipeline may still be running on server.`);
            setIsBackendDown(true);
            setTimeout(() => { setIsBackendDown(false); fetchReport(sym); }, 5000);
        };
    };

    const getStatusColor = (confidence) => {
        if (confidence >= 0.8) return 'var(--status-success)';
        if (confidence >= 0.5) return 'var(--status-warning)';
        return 'var(--status-danger)';
    };

    return (
        <div style={{ padding: '2.5rem', maxWidth: '1600px', margin: '0 auto' }}>

            {/* HEADER */}
            <header className="animate-slide-up" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
                    <div style={{ background: 'var(--accent-gradient)', padding: '14px', borderRadius: '14px', boxShadow: '0 0 20px rgba(0,242,254,0.2)' }}>
                        <Zap size={28} color="#000" />
                    </div>
                    <div>
                        <h1 style={{ fontSize: '2rem', fontWeight: '700', letterSpacing: '-0.5px', margin: 0, color: 'white' }}>Institutional Engine</h1>
                        <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0', fontSize: '0.95rem', fontWeight: '500' }}>Vol-Surface Analytics</p>
                    </div>
                </div>

                <div style={{ position: 'relative', width: '280px' }}>
                    <select
                        value={selectedSymbol}
                        onChange={(e) => handleSymbolChange(e.target.value)}
                        className="symbol-selector"
                    >
                        <option value="" disabled>Select Asset...</option>
                        {symbols.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                    <ChevronDown size={18} style={{ position: 'absolute', right: '16px', top: '15px', pointerEvents: 'none', color: 'var(--text-secondary)' }} />
                </div>
            </header>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '2.5rem' }}>

                {/* LEFT COLUMN: CONTROL & LOGS */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>

                    <div className="glass-panel animate-slide-up" style={{ padding: '2rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                            <h2 style={{ fontSize: '1.1rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <Shield size={20} color={report?.data ? 'var(--status-success)' : 'var(--text-secondary)'} />
                                Engine Status
                            </h2>
                            {report?.lastUpdated && (
                                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', background: 'rgba(255,255,255,0.05)', padding: '4px 10px', borderRadius: '20px' }}>
                                    Synced: {new Date(report.lastUpdated).toLocaleTimeString()}
                                </span>
                            )}
                        </div>

                        {report?.data ? (
                            <div className="animate-fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                <div className="metric-card">
                                    <div className="metric-label">Spot Reference</div>
                                    <div className="metric-value">{report.data.reference_price.toFixed(2)}</div>
                                </div>
                                <div className="metric-card">
                                    <div className="metric-label">Model Confidence</div>
                                    <div className="metric-value" style={{ color: getStatusColor(report.data.forecast_confidence) }}>
                                        {(report.data.forecast_confidence * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <div className="metric-card" style={{ borderLeft: '2px solid var(--status-danger)' }}>
                                    <div className="metric-label">Peak Excursion (90%)</div>
                                    <div className="metric-value" style={{ color: 'var(--status-danger)' }}>{report.data.projected_peak.toFixed(2)}</div>
                                </div>
                                <div className="metric-card" style={{ borderLeft: '2px solid var(--status-success)' }}>
                                    <div className="metric-label">Floor Excursion (90%)</div>
                                    <div className="metric-value" style={{ color: 'var(--status-success)' }}>{report.data.projected_bottom.toFixed(2)}</div>
                                </div>
                            </div>
                        ) : (
                            <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--text-secondary)' }}>
                                <BarChart2 size={36} style={{ margin: '0 auto 1rem auto', opacity: 0.3 }} />
                                <p>No verified artifacts located for {selectedSymbol}.</p>
                            </div>
                        )}

                        <button
                            className="btn-primary"
                            onClick={() => runInferenceForSymbol(selectedSymbol)}
                            disabled={isRunning || !selectedSymbol}
                            style={{ marginTop: '2rem' }}
                        >
                            {isRunning ? <RefreshCw size={18} className="animate-spin" /> : <Play size={18} fill="currentColor" />}
                            {isRunning ? 'CALCULATING SURFACES...' : 'EXECUTE PIPELINE'}
                        </button>
                    </div>

                    {/* DECOUPLED TERMINAL */}
                    <TerminalWindow ref={terminalRef} />
                </div>

                {/* RIGHT COLUMN: VISUALIZATIONS */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>

                    {/* PIPELINE STAGE TRACKER (Only visible when running) */}
                    {isRunning && (
                        <div className="glass-panel animate-fade-in" style={{ padding: '1.5rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div className="pipeline-stage">
                                <div className={`stage-indicator ${stages.vault_sync === 'running' ? 'animate-spin' : stages.vault_sync === null ? 'pending' : ''}`} style={{ color: stages.vault_sync === 'done' ? 'var(--status-success)' : 'var(--accent-cyan)' }}>
                                    {stages.vault_sync === 'done' ? <CheckCircle2 size={16} /> : stages.vault_sync === 'failed' ? <AlertTriangle size={16} color="var(--status-danger)"/> : stages.vault_sync === 'running' ? <RefreshCw size={14} /> : ''}
                                </div>
                                <span className={`stage-text ${stages.vault_sync ? 'active' : ''}`}>Vault Sync</span>
                            </div>
                            <div style={{ flex: 1, height: 1, background: 'var(--border-subtle)', margin: '0 15px' }} />

                            <div className="pipeline-stage">
                                <div className={`stage-indicator ${stages.training === 'running' ? 'animate-spin' : stages.training === null ? 'pending' : ''}`} style={{ color: stages.training === 'done' ? 'var(--status-success)' : 'var(--accent-cyan)' }}>
                                    {stages.training === 'done' ? <CheckCircle2 size={16} /> : stages.training === 'failed' ? <AlertTriangle size={16} color="var(--status-danger)"/> : stages.training === 'running' ? <RefreshCw size={14} /> : ''}
                                </div>
                                <span className={`stage-text ${stages.training ? 'active' : ''}`}>Engine Calibration</span>
                            </div>
                            <div style={{ flex: 1, height: 1, background: 'var(--border-subtle)', margin: '0 15px' }} />

                            <div className="pipeline-stage">
                                <div className={`stage-indicator ${stages.inference === 'running' ? 'animate-spin' : stages.inference === null ? 'pending' : ''}`} style={{ color: stages.inference === 'done' ? 'var(--status-success)' : 'var(--accent-cyan)' }}>
                                    {stages.inference === 'done' ? <CheckCircle2 size={16} /> : stages.inference === 'failed' ? <AlertTriangle size={16} color="var(--status-danger)"/> : stages.inference === 'running' ? <RefreshCw size={14} /> : ''}
                                </div>
                                <span className={`stage-text ${stages.inference ? 'active' : ''}`}>Live Inference</span>
                            </div>
                        </div>
                    )}

                    <div className="glass-panel animate-slide-up" style={{ padding: '2rem', flex: 1, display: 'flex', flexDirection: 'column', animationDelay: '0.1s' }}>
                        <div style={{ display: 'flex', gap: '2rem', marginBottom: '1.5rem' }}>
                            <button className={`tab-button ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>Forecast Visualization</button>
                            <button className={`tab-button ${activeTab === 'backtest' ? 'active' : ''}`} onClick={() => setActiveTab('backtest')}>Walk-Forward Analysis</button>
                        </div>

                        <div
                            className={`visualizer-container ${(activeTab === 'dashboard' ? report?.pngExists : report?.backtestExists) ? 'interactive' : ''}`}
                            style={{ flex: 1, minHeight: '500px' }}
                            onClick={() => {
                                if (activeTab === 'dashboard' && report?.pngExists) setIsFullscreen(true);
                                if (activeTab === 'backtest' && report?.backtestExists) setIsFullscreen(true);
                            }}
                        >
                            {activeTab === 'dashboard' ? (
                                report?.pngExists ? (
                                    <img
                                        src={`${API_URL}/image/${selectedSymbol}?type=dashboard&t=${report?.lastUpdated ? new Date(report.lastUpdated).getTime() : 0}`}
                                        alt="Live Forecast"
                                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                        className="animate-fade-in"
                                    />
                                ) : (
                                    <div style={{ color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                                        <Activity size={40} style={{ opacity: 0.2 }} className="animate-pulse" />
                                        <span>Awaiting artifacts...</span>
                                    </div>
                                )
                            ) : (
                                report?.backtestExists ? (
                                    <img
                                        src={`${API_URL}/image/${selectedSymbol}?type=backtest&t=${report?.lastUpdated ? new Date(report.lastUpdated).getTime() : 0}`}
                                        alt="Backtest Report"
                                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                        className="animate-fade-in"
                                    />
                                ) : (
                                    <div style={{ color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                                        <Activity size={40} style={{ opacity: 0.2 }} className="animate-pulse" />
                                        <span>Awaiting artifacts...</span>
                                    </div>
                                )
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* FULLSCREEN MODAL */}
            {isFullscreen && (
                <div
                    className="animate-fade-in"
                    onClick={() => setIsFullscreen(false)}
                    style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.95)', zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out', backdropFilter: 'blur(10px)' }}
                >
                    <img
                        src={`${API_URL}/image/${selectedSymbol}?type=${activeTab}&t=${report?.lastUpdated ? new Date(report.lastUpdated).getTime() : 0}`}
                        alt="Fullscreen"
                        style={{ maxWidth: '95vw', maxHeight: '95vh', objectFit: 'contain', filter: 'drop-shadow(0 0 40px rgba(0,242,254,0.1))' }}
                    />
                    <button style={{ position: 'absolute', top: '2rem', right: '3rem', background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white', cursor: 'pointer', padding: '10px', borderRadius: '50%', display: 'flex' }}>
                        <X size={24} />
                    </button>
                </div>
            )}

            {/* OFFLINE INDICATOR (Non-intrusive) */}
            {isBackendDown && (
                <div className="animate-fade-in" style={{ position: 'fixed', bottom: '2rem', right: '2rem', background: 'var(--status-danger)', color: 'white', padding: '12px 20px', borderRadius: '30px', display: 'flex', alignItems: 'center', gap: '10px', boxShadow: '0 4px 20px rgba(255,23,68,0.4)', fontWeight: '600', fontSize: '0.9rem', zIndex: 1000 }}>
                    <AlertTriangle size={18} /> Engine Offline
                </div>
            )}
        </div>
    );
}

export default App;
