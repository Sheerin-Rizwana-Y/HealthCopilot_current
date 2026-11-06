import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api, formatDate } from './api.js';

const TABS = [
  ['overview', 'Overview'],
  ['monitor', 'Live monitoring'],
  ['profile', 'Patient profile'],
  ['symptoms', 'Symptoms'],
  ['reports', 'Medical reports'],
  ['history', 'History & medicines'],
  ['alerts', 'Alerts'],
  ['emergency', 'Emergency response'],
  ['ai', 'Ask Lifora AI'],
  ['system', 'Device & system status'],
];

function Icon({ children }) {
  return <span className="icon" aria-hidden="true">{children}</span>;
}

function Login({ onLogin, busy, error }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  return (
    <main className="login-shell">
      <section className="login-card">
        <img className="brand-logo" src="/lifora-mark.svg" alt="Lifora" />
        <p className="eyebrow">SMART ELDERLY CARE PLATFORM</p>
        <h1>Welcome to Lifora</h1>
        <p className="muted">Bring wearable readings, caregiver observations and medical reports into one connected care workspace.</p>
        <form className="stack" onSubmit={(event) => { event.preventDefault(); onLogin(username, password); }}>
          <label>Username<input autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required placeholder="Enter your Lifora username" /></label>
          <label>Password<input autoComplete="current-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required /></label>
          {error && <p className="error-text">{error}</p>}
          <button className="primary full" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'} <span>→</span></button>
        </form>
        <div className="login-note"><Icon>ⓘ</Icon> Sign in with an account configured in the Lifora backend. Your credentials are verified by the server.</div>
      </section>
      <div className="login-side">
        <div className="side-orbit orbit-one" /><div className="side-orbit orbit-two" />
        <div className="side-copy"><span className="side-label">DESIGNED AROUND CARE</span><h2>More context.<br />Clearer care.</h2><p>Bring wearable readings, caregiver observations and medical reports into one view.</p></div>
        <div className="side-footer"><span className="status-dot" /> FastAPI backend <span>•</span> MongoDB-backed records</div>
      </div>
    </main>
  );
}

function MetricCard({ label, value, unit, icon, tone = '' }) {
  return <article className={`metric-card ${tone}`}><div className="metric-top"><span>{label}</span><span className="metric-icon">{icon}</span></div><div className="metric-value">{value ?? '—'}<small>{value !== null && value !== undefined ? unit : ''}</small></div><div className="metric-foot">Latest recorded value</div></article>;
}

function Empty({ title, detail }) {
  return <div className="empty-state"><span className="empty-symbol">○</span><strong>{title}</strong><p>{detail}</p></div>;
}

const blankEmergencyContacts = () => [1, 2, 3].map((priority) => ({ name: '', relationship: priority === 1 ? 'primary caregiver' : 'caregiver', phone: '', priority }));

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem('lifora_token') || '');
  const [user, setUser] = useState(null);
  const [loginBusy, setLoginBusy] = useState(false);
  const [loginError, setLoginError] = useState('');
  const [patients, setPatients] = useState([]);
  const [patientId, setPatientId] = useState('');
  const [tab, setTab] = useState('overview');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [dashboard, setDashboard] = useState(null);
  const [vitals, setVitals] = useState([]);
  const [vitalsUpdatedAt, setVitalsUpdatedAt] = useState(null);
  const [symptoms, setSymptoms] = useState([]);
  const [reports, setReports] = useState([]);
  const [history, setHistory] = useState([]);
  const [medications, setMedications] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [system, setSystem] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [question, setQuestion] = useState('Which symptoms appear in the patient reports, and what does the trusted medical reference say about them?');
  const [vitalForm, setVitalForm] = useState({ heart_rate_bpm: '82', spo2_percent: '98', temperature_c: '36.8', fall_detected: false, medicine_box_open: false, movement_detected: true });
  const [symptomForm, setSymptomForm] = useState({ name: '', severity: 'mild', onset_at: new Date().toISOString().slice(0, 16), notes: '' });
  const [historyForm, setHistoryForm] = useState({ category: 'condition', name: '', details: '', reported_on: new Date().toISOString().slice(0, 10) });
  const [medicationForm, setMedicationForm] = useState({ name: '', dosage_text: 'Not recorded', schedule_text: 'Not recorded', notes: '' });
  const [selectedFile, setSelectedFile] = useState(null);
  const [emergencyData, setEmergencyData] = useState(null);
  const [emergencyEvents, setEmergencyEvents] = useState([]);
  const [emergencyContacts, setEmergencyContacts] = useState([]);
  const [contactForm, setContactForm] = useState(blankEmergencyContacts);
  const [contactError, setContactError] = useState('');
  const [contactSaving, setContactSaving] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const audioContextRef = useRef(null);
  const [resolveConfirmed, setResolveConfirmed] = useState({});
  const [resolutionNotes, setResolutionNotes] = useState({});

  const selectedPatient = useMemo(() => patients.find((p) => p.patient_id === patientId), [patients, patientId]);
  const activeEmergencyEvent = emergencyData?.active_events?.[0] || null;
  const activeAlarmKey = activeEmergencyEvent
    ? `${activeEmergencyEvent.event_id}:${activeEmergencyEvent.state}:${activeEmergencyEvent.local_alarm_state}:${activeEmergencyEvent.severity}`
    : '';

  const refreshPatientData = useCallback(async (id = patientId, authToken = token) => {
    if (!id || !authToken) return;
    setError('');
    const paths = [
      `/patients/${id}/dashboard`, `/patients/${id}/vitals?limit=30`, `/patients/${id}/symptoms`,
      `/patients/${id}/reports`, `/patients/${id}/history`, `/patients/${id}/medications`, `/patients/${id}/alerts`,
    ];
    const [dash, vitalRows, symptomRows, reportRows, historyRows, medRows, alertRows] = await Promise.all(
      paths.map((path) => api(path, { token: authToken })),
    );
    setDashboard(dash); setVitals(vitalRows); setVitalsUpdatedAt(new Date().toISOString()); setSymptoms(symptomRows); setReports(reportRows);
    setHistory(historyRows); setMedications(medRows); setAlerts(alertRows);
  }, [patientId, token]);

  const refreshEmergencySnapshot = useCallback(async (id = patientId, authToken = token) => {
    if (!id || !authToken) return;
    const [snapshot, events] = await Promise.all([
      api(`/patients/${id}/emergency`, { token: authToken }),
      api(`/patients/${id}/emergency-events?limit=50`, { token: authToken }),
    ]);
    setEmergencyData(snapshot);
    setEmergencyEvents(events);
  }, [patientId, token]);

  const loadEmergencyContacts = useCallback(async (id = patientId, authToken = token) => {
    if (!id || !authToken) return;
    try {
      const rows = await api(`/patients/${id}/emergency-contacts`, { token: authToken });
      setEmergencyContacts(rows);
      setContactForm([...rows.map((row) => ({ ...row })), ...blankEmergencyContacts()].slice(0, 3));
      setContactError('');
    } catch (e) {
      setEmergencyContacts([]);
      setContactForm(blankEmergencyContacts());
      setContactError(e.message);
    }
  }, [patientId, token]);

  useEffect(() => {
    if (!token || !patientId) return;
    let cancelled = false;
    async function pollEmergency() {
      try {
        const [snapshot, events] = await Promise.all([
          api(`/patients/${patientId}/emergency`, { token }),
          api(`/patients/${patientId}/emergency-events?limit=50`, { token }),
        ]);
        if (!cancelled) { setEmergencyData(snapshot); setEmergencyEvents(events); }
      } catch (e) {
        if (!cancelled) setContactError((current) => current || `Emergency status could not refresh: ${e.message}`);
      }
    }
    pollEmergency();
    const timer = window.setInterval(pollEmergency, 5000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [token, patientId]);

  useEffect(() => {
    if (!token || !patientId || tab !== 'monitor') return undefined;
    let cancelled = false;
    async function pollVitals() {
      try {
        const rows = await api(`/patients/${patientId}/vitals?limit=30`, { token });
        if (!cancelled) {
          setVitals(rows);
          setVitalsUpdatedAt(new Date().toISOString());
        }
      } catch (e) {
        if (!cancelled) setError((current) => current || `Sensor readings could not refresh: ${e.message}`);
      }
    }
    pollVitals();
    const timer = window.setInterval(pollVitals, 10000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [token, patientId, tab]);

  useEffect(() => {
    if (token && patientId && tab === 'emergency') loadEmergencyContacts(patientId, token);
  }, [token, patientId, tab, loadEmergencyContacts]);

  useEffect(() => {
    const event = emergencyData?.active_events?.[0];
    const context = audioContextRef.current;
    if (!audioEnabled || !event || event.local_alarm_state !== 'active' || !context) return undefined;
    const critical = event.severity === 'critical';
    const beep = () => {
      if (context.state !== 'running') return;
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = 'square';
      oscillator.frequency.value = critical ? 880 : 560;
      gain.gain.setValueAtTime(0.001, context.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.12, context.currentTime + 0.025);
      gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.22);
      oscillator.connect(gain); gain.connect(context.destination);
      oscillator.start(); oscillator.stop(context.currentTime + 0.23);
    };
    beep();
    const timer = window.setInterval(beep, critical ? 950 : 1700);
    return () => window.clearInterval(timer);
  }, [audioEnabled, activeAlarmKey]);

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      if (!token) return;
      try {
        const [me, patientRows] = await Promise.all([api('/auth/me', { token }), api('/patients', { token })]);
        if (cancelled) return;
        setUser(me); setPatients(patientRows);
        setPatientId((current) => current && patientRows.some((p) => p.patient_id === current) ? current : patientRows[0]?.patient_id || '');
      } catch (e) {
        localStorage.removeItem('lifora_token'); setToken(''); setUser(null); setLoginError(e.message);
      }
    }
    bootstrap();
    return () => { cancelled = true; };
  }, [token]);

  useEffect(() => {
    if (!token || !patientId) return;
    let cancelled = false;
    setBusy(true);
    refreshPatientData(patientId, token).catch((e) => { if (!cancelled) setError(e.message); }).finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [token, patientId, refreshPatientData]);

  async function handleLogin(username, password) {
    setLoginBusy(true); setLoginError('');
    try {
      const result = await api('/auth/login', { method: 'POST', body: { username, password } });
      localStorage.setItem('lifora_token', result.access_token);
      setToken(result.access_token); setUser(result.user); setLoginError('');
    } catch (e) { setLoginError(e.message); }
    finally { setLoginBusy(false); }
  }

  function signOut() {
    localStorage.removeItem('lifora_token'); setToken(''); setUser(null); setPatients([]); setPatientId('');
    setDashboard(null); setExplanation(null); setNotice(''); setError(''); setEmergencyData(null); setEmergencyEvents([]); setEmergencyContacts([]); setAudioEnabled(false);
  }

  async function action(work, successMessage) {
    setBusy(true); setError(''); setNotice('');
    try { await work(); if (successMessage) setNotice(successMessage); if (patientId) await refreshPatientData(patientId, token); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function refreshVitalsNow() {
    if (!patientId || !token) return;
    try {
      const rows = await api(`/patients/${patientId}/vitals?limit=30`, { token });
      setVitals(rows);
      setVitalsUpdatedAt(new Date().toISOString());
    } catch (e) {
      setError(e.message);
    }
  }

  async function submitVital(event) {
    event.preventDefault();
    const body = {
      heart_rate_bpm: vitalForm.heart_rate_bpm === '' ? null : Number(vitalForm.heart_rate_bpm),
      spo2_percent: vitalForm.spo2_percent === '' ? null : Number(vitalForm.spo2_percent),
      temperature_c: vitalForm.temperature_c === '' ? null : Number(vitalForm.temperature_c),
      fall_detected: vitalForm.fall_detected,
      medicine_box_open: vitalForm.medicine_box_open,
      movement_detected: vitalForm.movement_detected,
      motion_event: vitalForm.movement_detected ? 'normal_motion' : 'still',
      signal_quality: 'good',
      source: 'caregiver_manual',
    };
    await action(() => api(`/patients/${patientId}/vitals`, { token, method: 'POST', body }), 'Reading saved. Rule alerts are evaluated separately from the LLM.');
  }

  async function submitSymptom(event) {
    event.preventDefault();
    const body = { ...symptomForm, onset_at: new Date(symptomForm.onset_at).toISOString() };
    await action(async () => { await api(`/patients/${patientId}/symptoms`, { token, method: 'POST', body }); setSymptomForm({ name: '', severity: 'mild', onset_at: new Date().toISOString().slice(0, 16), notes: '' }); }, 'Caregiver symptom saved as unverified.');
  }

  async function submitHistory(event) {
    event.preventDefault();
    await action(async () => { await api(`/patients/${patientId}/history`, { token, method: 'POST', body: historyForm }); setHistoryForm({ category: 'condition', name: '', details: '', reported_on: new Date().toISOString().slice(0, 10) }); }, 'History entry saved as unverified.');
  }

  async function submitMedication(event) {
    event.preventDefault();
    await action(async () => { await api(`/patients/${patientId}/medications`, { token, method: 'POST', body: medicationForm }); setMedicationForm({ name: '', dosage_text: 'Not recorded', schedule_text: 'Not recorded', notes: '' }); }, 'Medication record saved. This does not verify a prescription.');
  }

  async function submitReport(event) {
    event.preventDefault();
    if (!selectedFile) { setError('Choose a PDF first.'); return; }
    const form = new FormData(); form.append('file', selectedFile);
    await action(async () => { const result = await api(`/patients/${patientId}/reports`, { token, method: 'POST', body: form }); setSelectedFile(null); if (result.report?.index_status === 'failed') setNotice(`Report stored but indexing failed: ${result.report.index_error || 'check backend logs'}`); else setNotice(result.message || 'Report uploaded and indexed.'); }, null);
  }

  async function submitExplanation(event) {
    event.preventDefault();
    setBusy(true); setError(''); setNotice(''); setExplanation(null);
    try {
      const result = await api(`/patients/${patientId}/explain`, { token, method: 'POST', body: { question, top_k: 3 } });
      setExplanation(result);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function acknowledge(alertId) {
    await action(() => api(`/patients/${patientId}/alerts/${alertId}/acknowledge`, { token, method: 'POST' }), 'Alert acknowledged.');
  }

  async function recordMedicationEvent(medicationId, eventName) {
    await action(() => api(`/patients/${patientId}/medications/${medicationId}/events`, { token, method: 'POST', body: { event: eventName, notes: 'Recorded from caregiver dashboard.' } }), 'Medication event recorded.');
  }

  async function loadSystemStatus() {
    setBusy(true); setError('');
    try { setSystem(await api('/system/status', { token })); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  function enableBrowserAlarm() {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) { setError('This browser does not support browser audio alerts. The hardware buzzer adapter is separate.'); return; }
      if (!audioContextRef.current) audioContextRef.current = new AudioContextClass();
      audioContextRef.current.resume();
      setAudioEnabled(true);
      setNotice('Browser alarm enabled for this open dashboard session. Keep the page open; this is not a substitute for the device buzzer.');
    } catch (e) { setError(`Could not enable browser sound: ${e.message}`); }
  }

  async function saveEmergencyContacts(event) {
    event.preventDefault();
    const filled = contactForm.filter((row) => row.name.trim() || row.phone.trim());
    if (filled.some((row) => !row.name.trim() || !row.phone.trim())) {
      setContactError('For each contact row you use, enter both a name and a phone number.');
      return;
    }
    const contacts = filled.map((row, index) => ({
      name: row.name.trim(), relationship: row.relationship.trim() || 'caregiver',
      phone: row.phone.trim(), priority: index + 1,
    }));
    setContactSaving(true); setContactError(''); setError(''); setNotice('');
    try {
      const result = await api(`/patients/${patientId}/emergency-contacts`, { token, method: 'PUT', body: { contacts } });
      setNotice(`Saved ${result.contact_count} emergency contact(s). Phone numbers are encrypted in MongoDB.`);
      await loadEmergencyContacts(patientId, token);
      await refreshEmergencySnapshot(patientId, token);
    } catch (e) { setContactError(e.message); }
    finally { setContactSaving(false); }
  }

  async function emergencyEventAction(eventId, actionName) {
    setBusy(true); setError(''); setNotice('');
    try {
      if (actionName === 'respond') {
        await api(`/patients/${patientId}/emergency-events/${eventId}/respond`, { token, method: 'POST', body: { person_is_responsive: true } });
        setNotice('Responsiveness recorded. The event remains unresolved and monitoring continues.');
      } else if (actionName === 'acknowledge') {
        await api(`/patients/${patientId}/emergency-events/${eventId}/acknowledge`, { token, method: 'POST' });
        setNotice('Alert acknowledged. This does not resolve the event; monitoring continues.');
      } else if (actionName === 'resolve') {
        const notes = (resolutionNotes[eventId] || '').trim();
        if (!resolveConfirmed[eventId]) throw new Error('Tick the safety-confirmation checkbox after physically verifying the person.');
        if (notes.length < 3) throw new Error('Add a short resolution note before resolving the event.');
        await api(`/patients/${patientId}/emergency-events/${eventId}/resolve`, { token, method: 'POST', body: { safety_confirmed: true, notes } });
        setNotice('Event marked resolved after caregiver confirmation.');
        setResolveConfirmed((current) => ({ ...current, [eventId]: false }));
      }
      await refreshEmergencySnapshot(patientId, token);
      await refreshPatientData(patientId, token);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  useEffect(() => { if (token && tab === 'system') loadSystemStatus(); }, [token, tab]);

  if (!token) return <Login onLogin={handleLogin} busy={loginBusy} error={loginError} />;

  const latest = dashboard?.latest_vitals;
  const openAlerts = alerts.filter((item) => !item.acknowledged);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row"><div className="brand-mark small"><img src="/lifora-mark.svg" alt="" /></div><div><strong>Lifora</strong><span>{user?.role === 'admin' ? 'Administrator workspace' : 'Care workspace'}</span></div></div>
        <div className="side-caption">WORKSPACE</div>
        <div className="patient-select-wrap"><label htmlFor="patient-select">Patient profile</label><select id="patient-select" value={patientId} onChange={(e) => setPatientId(e.target.value)}>{patients.map((p) => <option key={p.patient_id} value={p.patient_id}>{p.display_name} · {p.patient_id}</option>)}</select></div>
        <nav className="side-nav">{TABS.map(([key, label], index) => <button key={key} className={tab === key ? 'nav-item active' : 'nav-item'} onClick={() => setTab(key)}><span className="nav-number">{String(index + 1).padStart(2, '0')}</span>{label}{tab === key && <span className="nav-arrow">↗</span>}</button>)}</nav>
        <div className="sidebar-bottom"><div className="privacy-chip"><span className="status-dot" /><div><strong>Authenticated API</strong><small>Server-verified session</small></div></div><button className="signout" onClick={signOut}><Icon>↗</Icon> Sign out</button></div>
      </aside>

      <main className="main-area">
        <header className="topbar"><div><p className="eyebrow">LIFORA / {(user?.role || 'caregiver').toUpperCase()} PORTAL</p><h1>{TABS.find(([key]) => key === tab)?.[1] || 'Overview'}</h1></div><div className="topbar-right"><span className="connection-pill"><span className="status-dot" /> API session active</span><div className="avatar">{(user?.username || 'C').slice(0, 1).toUpperCase()}</div><div className="user-meta"><strong>{user?.username || 'Caregiver'}</strong><span>{user?.role || 'caregiver'}</span></div></div></header>

        {activeEmergencyEvent && activeEmergencyEvent.local_alarm_state === 'active' && <div className={`emergency-global-banner ${activeEmergencyEvent.severity === 'critical' ? 'critical' : 'warning'}`} role="alert"><div><strong>{activeEmergencyEvent.severity === 'critical' ? 'CRITICAL EMERGENCY ALERT' : 'HEALTH WARNING'}</strong><span>{activeEmergencyEvent.event_type.replaceAll('_', ' ')} · {activeEmergencyEvent.reasons?.[0] || 'Review the active event.'}</span></div><button className="secondary small-button" onClick={() => setTab('emergency')}>Review alert</button>{!audioEnabled && <button className="primary small-button" onClick={enableBrowserAlarm}>Enable sound</button>}</div>}
        {error && <div className="banner error-banner"><Icon>!</Icon><span>{error}</span><button onClick={() => setError('')}>×</button></div>}
        {notice && <div className="banner success-banner"><Icon>✓</Icon><span>{notice}</span><button onClick={() => setNotice('')}>×</button></div>}
        {busy && <div className="loading-line"><span /></div>}

        {tab === 'overview' && <>
          <section className="welcome-row"><div><h2>{selectedPatient?.display_name || 'Patient overview'}</h2><p>Current status and recent observations <span className="subtle-separator">/</span> <span className="muted">{selectedPatient?.synthetic ? 'Synthetic demo profile' : 'Caregiver-managed profile'}</span></p></div><div className="updated-label">Updated {formatDate(latest?.recorded_at)}</div></section>
          <section className="metric-grid">
            <MetricCard label="Heart rate" value={latest?.heart_rate_bpm} unit=" bpm" icon="♡" />
            <MetricCard label="Blood oxygen" value={latest?.spo2_percent} unit="%" icon="◉" tone="teal" />
            <MetricCard label="Temperature" value={latest?.temperature_c} unit=" °C" icon="⌁" tone="sand" />
            <MetricCard label="Open alerts" value={dashboard?.unacknowledged_alert_count ?? 0} unit="" icon="!" tone={openAlerts.length ? 'rose' : ''} />
          </section>
          <section className="two-column">
            <div className="panel"><div className="panel-heading"><div><p className="eyebrow">DEVICE / CAREGIVER INPUT</p><h3>Add a reading</h3></div><span className="panel-icon">⌁</span></div><p className="panel-description">Manual test input. The ESP32 can post to the protected device-ingest endpoint.</p>
              <form className="form-grid" onSubmit={submitVital}>
                <label>Heart rate (bpm)<input type="number" min="20" max="250" value={vitalForm.heart_rate_bpm} onChange={(e) => setVitalForm({ ...vitalForm, heart_rate_bpm: e.target.value })} /></label>
                <label>SpO₂ (%)<input type="number" min="50" max="100" step="0.1" value={vitalForm.spo2_percent} onChange={(e) => setVitalForm({ ...vitalForm, spo2_percent: e.target.value })} /></label>
                <label>Temperature (°C)<input type="number" min="25" max="45" step="0.1" value={vitalForm.temperature_c} onChange={(e) => setVitalForm({ ...vitalForm, temperature_c: e.target.value })} /></label>
                <div className="check-stack"><label className="check-label"><input type="checkbox" checked={vitalForm.fall_detected} onChange={(e) => setVitalForm({ ...vitalForm, fall_detected: e.target.checked })} /> Existing fall-model event (test)</label><label className="check-label"><input type="checkbox" checked={vitalForm.movement_detected} onChange={(e) => setVitalForm({ ...vitalForm, movement_detected: e.target.checked })} /> Movement detected</label><label className="check-label"><input type="checkbox" checked={vitalForm.medicine_box_open} onChange={(e) => setVitalForm({ ...vitalForm, medicine_box_open: e.target.checked })} /> Medicine box opened</label></div>
                <button className="primary" disabled={busy}>Save reading <span>→</span></button>
              </form><p className="micro-note">Rule thresholds are illustrative prototype settings, not clinically validated alarms.</p>
            </div>
            <div className="panel alerts-panel"><div className="panel-heading"><div><p className="eyebrow">INDEPENDENT RULE ENGINE</p><h3>Alerts to review <span className="count-bubble">{openAlerts.length}</span></h3></div><span className="panel-icon alert-icon">!</span></div><p className="panel-description">Alerts do not depend on the LLM and cannot be dismissed by an AI explanation.</p>
              {openAlerts.length === 0 ? <Empty title="No open prototype alerts" detail="Device and manual readings are checked against the current demo rules." /> : <div className="alert-list">{openAlerts.slice(0, 5).map((item) => <div className="alert-row" key={item.id}><div className={`severity-mark ${item.severity}`}>!</div><div className="alert-copy"><strong>{item.title}</strong><p>{item.message}</p><small>{formatDate(item.created_at)}</small></div><button className="icon-button" title="Acknowledge" onClick={() => acknowledge(item.id)}>✓</button></div>)}</div>}
            </div>
          </section>
          <section className="panel ai-cta">
            <div>
              <p className="eyebrow">RAG + LLM</p>
              <h3>Ask Lifora AI</h3>
              <p className="panel-description">Ask a question about the patient's recorded symptoms, history and uploaded reports. Answers use retrieved evidence and citations where available.</p>
            </div>
            <button className="primary" onClick={() => setTab('ai')}>Open AI assistant <span>→</span></button>
          </section>
        </>}

        {tab === 'alerts' && <section className="panel">
          <div className="panel-heading"><div><p className="eyebrow">RULE-BASED NOTIFICATIONS</p><h3>Alerts and review queue <span className="count-bubble">{openAlerts.length} open</span></h3></div></div>
          <p className="panel-description">These alerts are generated by the backend's independent rule engine. RAG explanations do not create, suppress or resolve alerts.</p>
          {alerts.length === 0 ? <Empty title="No alerts recorded" detail="Alerts will appear when readings meet configured prototype rules." /> :
            <div className="alert-list">{alerts.map((item) => <div className="alert-row" key={item.id || item.alert_id}>
              <div className={`severity-mark ${item.severity || 'medium'}`}>!</div>
              <div className="alert-copy"><strong>{item.title || item.rule || 'Health alert'}</strong><p>{item.message || item.description || 'Review the recorded alert.'}</p><small>{formatDate(item.created_at || item.recorded_at)} · {item.acknowledged ? 'Acknowledged' : 'Awaiting review'}</small></div>
              {!item.acknowledged && <button className="icon-button" title="Acknowledge alert" onClick={() => acknowledge(item.alert_id || item.id)}>✓</button>}
            </div>)}</div>}
        </section>}

        {tab === 'emergency' && <>
          <section className="emergency-summary-grid">
            <div className={`panel emergency-status-card status-${emergencyData?.monitoring_status || 'loading'}`}>
              <div className="panel-heading"><div><p className="eyebrow">CONTINUOUS MONITORING</p><h3>Emergency status</h3></div><span className={`emergency-status-pill ${activeEmergencyEvent?.severity || emergencyData?.monitoring_status || 'unknown'}`}>{(emergencyData?.monitoring_status || 'loading').replaceAll('_', ' ')}</span></div>
              {activeEmergencyEvent ? <><p className="emergency-event-title">{activeEmergencyEvent.event_type.replaceAll('_', ' ')}</p><p className="panel-description">{activeEmergencyEvent.reasons?.join(' · ') || 'An unresolved event needs review.'}</p><div className="emergency-meta"><span>Started {formatDate(activeEmergencyEvent.created_at)}</span><span>Local alarm: {activeEmergencyEvent.local_alarm_state || 'silent'}</span><span>Notifications: {(activeEmergencyEvent.notification_status || 'unknown').replaceAll('_', ' ')}</span></div></> : <p className="panel-description">{emergencyData?.monitoring_status === 'normal_verified' ? 'Current readings and movement are within configured prototype ranges.' : 'Safety cannot be marked normal until sensor readings and movement evidence are sufficient.'}</p>}
              <div className="emergency-actions"><button className="primary" onClick={enableBrowserAlarm}>Enable browser sound</button><button className="secondary" onClick={() => refreshEmergencySnapshot().catch((e) => setError(e.message))}>Refresh status ↻</button></div>
              <p className="micro-note">Browser audio works only while this page is open and sound is enabled. The ESP32 buzzer/LED must be connected by its firmware using the device alert-state endpoint.</p>
            </div>
            <div className="panel"><div className="panel-heading"><div><p className="eyebrow">MONITORING POLICY</p><h3>Active configuration</h3></div></div>
              {emergencyData?.configuration ? <div className="policy-list"><div><span>Inactivity threshold</span><strong>{emergencyData.configuration.inactivity_threshold_seconds}s</strong></div><div><span>Sensor data stale timeout</span><strong>{emergencyData.configuration.sensor_stale_after_seconds}s</strong></div><div><span>Vital trend window</span><strong>{emergencyData.configuration.trend_window_seconds}s</strong></div><div><span>Check-in response timeout</span><strong>{emergencyData.configuration.inactivity_response_timeout_seconds}s</strong></div><div><span>Critical alarm / local snooze</span><strong>{emergencyData.configuration.local_alarm_active_seconds}s / {emergencyData.configuration.local_alarm_snooze_seconds}s</strong></div><div><span>Persistent vital threshold</span><strong>{emergencyData.configuration.vital_persistence_seconds}s</strong></div><div><span>Twilio SMS / calls</span><strong>{emergencyData.configuration.sms_enabled ? 'SMS on' : 'SMS off'} · {emergencyData.configuration.calls_enabled ? 'calls on' : 'calls off'}</strong></div><div><span>Twilio credentials</span><strong>{emergencyData.configuration.twilio_credentials_configured ? 'configured' : 'missing'}</strong></div><div><span>Contact encryption key</span><strong>{emergencyData.configuration.contact_encryption_configured ? 'configured' : 'not configured'}</strong></div></div> : <Empty title="Waiting for monitoring status" detail="Status refreshes automatically every five seconds." />}
            </div>
          </section>

          <section className="page-grid emergency-page-grid">
            <div className="panel"><div className="panel-heading"><div><p className="eyebrow">CONTACT CONFIGURATION</p><h3>Emergency contacts</h3></div></div>
              <p className="panel-description">Use E.164 format, including the plus sign and country code. Phone numbers are encrypted in MongoDB; configure the encryption key before saving. Twilio delivery is disabled by default.</p>
              {contactError && <div className="inline-error">{contactError}</div>}
              <form className="stack" onSubmit={saveEmergencyContacts}>
                {contactForm.map((contact, index) => <div className="contact-editor" key={contact.contact_id || `contact-${index}`}><div className="contact-editor-title"><strong>{index === 0 ? 'Primary contact' : `Secondary contact ${index + 1}`}</strong><span>Priority {index + 1}</span></div><label>Name<input value={contact.name || ''} onChange={(e) => setContactForm((rows) => rows.map((row, i) => i === index ? { ...row, name: e.target.value } : row))} maxLength={100} placeholder="Caregiver name" /></label><label>Relationship<input value={contact.relationship || ''} onChange={(e) => setContactForm((rows) => rows.map((row, i) => i === index ? { ...row, relationship: e.target.value } : row))} maxLength={50} placeholder="e.g. family caregiver" /></label><label>Phone number<input type="tel" value={contact.phone || ''} onChange={(e) => setContactForm((rows) => rows.map((row, i) => i === index ? { ...row, phone: e.target.value } : row))} maxLength={16} placeholder="+countrycode and number" /></label></div>)}
                <button className="primary" disabled={contactSaving || busy}>{contactSaving ? 'Saving contacts…' : 'Save encrypted contacts'} <span>→</span></button>
              </form>
              <p className="micro-note">Do not enter real numbers for a demo unless you have consent and intentionally configured a verified Twilio test workflow. Numbers are never written to application logs.</p>
            </div>
            <div className="panel"><div className="panel-heading"><div><p className="eyebrow">UNRESOLVED EVENTS</p><h3>Emergency queue <span className="count-bubble">{emergencyData?.active_events?.length || 0}</span></h3></div></div>
              {!emergencyData ? <Empty title="Loading emergency state" detail="The API status is being retrieved." /> : (emergencyData.active_events || []).length === 0 ? <Empty title="No unresolved emergency events" detail={emergencyData.monitoring_status === 'normal_verified' ? 'Latest available readings are verified within the configured prototype ranges.' : 'No active event is recorded, but missing movement or vital data is not considered proof of safety.'} /> : <div className="emergency-event-list">{emergencyData.active_events.map((event) => <article className={`emergency-event-card ${event.severity}`} key={event.event_id}><div className="emergency-event-head"><div><strong>{event.event_type.replaceAll('_', ' ')}</strong><small>{formatDate(event.created_at)} · {event.state.replaceAll('_', ' ')}</small></div><span className={`emergency-status-pill ${event.severity}`}>{event.severity}</span></div><ul>{(event.reasons || []).map((reason, i) => <li key={`${event.event_id}-reason-${i}`}>{reason}</li>)}</ul><p className="micro-note">Acknowledgment does not resolve this event. Verify the person's safety before resolving.</p><div className="event-action-row">{(event.severity === 'warning' || event.event_type.includes('inactivity')) && !event.person_responded && <button className="secondary small-button" disabled={busy} onClick={() => emergencyEventAction(event.event_id, 'respond')}>Record person check-in</button>}{!event.acknowledged && <button className="secondary small-button" disabled={busy} onClick={() => emergencyEventAction(event.event_id, 'acknowledge')}>Caregiver acknowledge</button>}</div><div className="resolve-controls"><label className="check-label"><input type="checkbox" checked={!!resolveConfirmed[event.event_id]} onChange={(e) => setResolveConfirmed((current) => ({ ...current, [event.event_id]: e.target.checked }))} /> I have physically verified the person's safety</label><label>Resolution note<textarea rows="2" value={resolutionNotes[event.event_id] || ''} onChange={(e) => setResolutionNotes((current) => ({ ...current, [event.event_id]: e.target.value }))} maxLength={300} placeholder="Describe the safety check performed" /></label><button className="primary small-button" disabled={busy || !resolveConfirmed[event.event_id] || (resolutionNotes[event.event_id] || '').trim().length < 3} onClick={() => emergencyEventAction(event.event_id, 'resolve')}>Resolve after verification</button></div><details><summary>Event timeline</summary><div className="timeline-list">{(event.timeline || []).map((item, i) => <div className="timeline-item" key={`${event.event_id}-timeline-${i}`}><span className="timeline-dot" /><div><strong>{item.kind?.replaceAll('_', ' ')}</strong><p>{item.note}</p><small>{formatDate(item.at)}</small></div></div>)}</div></details></article>)}</div>}
            </div>
          </section>
          <section className="panel system-panel"><div className="panel-heading"><div><p className="eyebrow">AUDIT HISTORY</p><h3>Recent emergency events</h3></div></div>{emergencyEvents.filter((event) => event.status === 'resolved').length === 0 ? <Empty title="No resolved events yet" detail="Resolved emergency events and their audit trails will appear here." /> : <div className="emergency-event-list">{emergencyEvents.filter((event) => event.status === 'resolved').slice(0, 10).map((event) => <article className="history-item" key={event.event_id}><div className="history-symbol">✓</div><div className="history-copy"><strong>{event.event_type.replaceAll('_', ' ')}</strong><span>Resolved · {formatDate(event.resolved_at)}</span><p>{event.resolution_note || 'No resolution note recorded.'}</p></div></article>)}</div>}</section>
        </>}

        {tab === 'monitor' && <>
          <section className="welcome-row">
            <div><h2>Live sensor monitoring</h2><p>Recent readings retrieved from the Lifora API. This page refreshes every 10 seconds while open.</p></div>
            <div className="updated-label">Last refresh {formatDate(vitalsUpdatedAt)}</div>
          </section>
          <section className="metric-grid">
            <MetricCard label="Heart rate" value={vitals[0]?.heart_rate_bpm} unit=" bpm" icon="♡" />
            <MetricCard label="Blood oxygen" value={vitals[0]?.spo2_percent} unit="%" icon="◉" tone="teal" />
            <MetricCard label="Temperature" value={vitals[0]?.temperature_c} unit=" °C" icon="⌁" tone="sand" />
            <MetricCard label="Recent readings" value={vitals.length} unit="" icon="▤" />
          </section>
          <section className="panel">
            <div className="panel-heading">
              <div><p className="eyebrow">RECENT TELEMETRY</p><h3>Sensor reading history</h3></div>
              <button className="secondary" onClick={refreshVitalsNow}>Refresh now ↻</button>
            </div>
            {vitals.length === 0 ? <Empty title="No sensor readings available" detail="Connect the ESP32 or record a test reading from the Overview page." /> :
              <div className="table-wrap"><table className="data-table">
                <thead><tr><th>Recorded at</th><th>Heart rate</th><th>SpO₂</th><th>Temperature</th><th>Fall flag</th><th>Movement</th><th>Source</th></tr></thead>
                <tbody>{vitals.map((row, index) => <tr key={row.id || row._id || `${row.recorded_at}-${index}`}>
                  <td>{formatDate(row.recorded_at)}</td><td>{row.heart_rate_bpm ?? '—'} bpm</td><td>{row.spo2_percent ?? '—'}%</td><td>{row.temperature_c ?? '—'} °C</td>
                  <td><span className={`status-badge ${row.fall_detected ? 'failed' : 'indexed'}`}>{row.fall_detected ? 'Fall flag' : 'No fall flag'}</span></td>
                  <td>{row.movement_detected === true ? 'Detected' : row.movement_detected === false ? 'Not detected' : '—'}</td><td>{row.source || 'unknown'}</td>
                </tr>)}</tbody>
              </table></div>}
            <p className="micro-note">Readings are shown as received from the backend. Missing values are displayed as unavailable, not zero. These measurements are not a diagnosis.</p>
          </section>
        </>}

        {tab === 'profile' && <section className="page-grid profile-grid">
          <div className="panel">
            <div className="panel-heading"><div><p className="eyebrow">PATIENT RECORD</p><h3>{selectedPatient?.display_name || 'Patient profile'}</h3></div><span className="panel-icon">♙</span></div>
            {selectedPatient ? <div className="policy-list">
              <div><span>Patient ID</span><strong>{selectedPatient.patient_id}</strong></div>
              <div><span>Record status</span><strong>{selectedPatient.status || 'Not recorded'}</strong></div>
              <div><span>Data classification</span><strong>{selectedPatient.synthetic ? 'Synthetic demo record' : 'Non-synthetic record'}</strong></div>
              <div><span>Created</span><strong>{formatDate(selectedPatient.created_at)}</strong></div>
            </div> : <Empty title="No patient selected" detail="Select a patient record from the sidebar." />}
            <p className="micro-note">This backend version stores a limited patient profile. Age, gender, address and editable demographic fields are not exposed by the current API, so they are not fabricated here.</p>
          </div>
          <div className="panel">
            <div className="panel-heading"><div><p className="eyebrow">RECORDED CONTEXT</p><h3>Patient record summary</h3></div></div>
            <div className="stat-card"><div><small>Symptoms</small><strong>{dashboard?.symptom_count ?? symptoms.length}</strong><em>Caregiver-reported entries</em></div></div>
            <div className="stat-card"><div><small>Medical reports</small><strong>{dashboard?.report_count ?? reports.length}</strong><em>Uploaded documents</em></div></div>
            <div className="stat-card"><div><small>History entries</small><strong>{dashboard?.history_count ?? history.length}</strong><em>Unverified structured context</em></div></div>
          </div>
        </section>}

        {tab === 'symptoms' && <section className="page-grid"><div className="panel"><div className="panel-heading"><div><p className="eyebrow">CAREGIVER OBSERVATION</p><h3>Record a symptom</h3></div></div><form className="stack" onSubmit={submitSymptom}><label>Symptom<input value={symptomForm.name} onChange={(e) => setSymptomForm({ ...symptomForm, name: e.target.value })} required maxLength={100} placeholder="e.g. dizziness" /></label><label>Severity<select value={symptomForm.severity} onChange={(e) => setSymptomForm({ ...symptomForm, severity: e.target.value })}><option value="mild">Mild</option><option value="moderate">Moderate</option><option value="severe">Severe</option></select></label><label>Onset date and time<input type="datetime-local" value={symptomForm.onset_at} onChange={(e) => setSymptomForm({ ...symptomForm, onset_at: e.target.value })} required /></label><label>Notes<textarea rows="3" value={symptomForm.notes} onChange={(e) => setSymptomForm({ ...symptomForm, notes: e.target.value })} maxLength={500} placeholder="Optional context, not a diagnosis" /></label><button className="primary" disabled={busy}>Save symptom <span>→</span></button></form><p className="micro-note">Every entry is marked caregiver-reported and unverified.</p></div><div className="panel"><div className="panel-heading"><div><p className="eyebrow">RECENT RECORDS</p><h3>Symptom timeline <span className="count-bubble">{symptoms.length}</span></h3></div></div>{symptoms.length === 0 ? <Empty title="No symptoms recorded" detail="Add a caregiver observation to start the timeline." /> : <div className="timeline-list">{symptoms.map((item) => <article className="timeline-item" key={item.id}><span className={`timeline-dot ${item.severity}`} /><div><div className="timeline-title"><strong>{item.name}</strong><span className={`severity-pill ${item.severity}`}>{item.severity}</span></div><p>{item.notes || 'No additional notes.'}</p><small>{formatDate(item.onset_at)} · {item.verification_status || 'unverified'}</small></div></article>)}</div>}</div></section>}

        {tab === 'reports' && <section className="page-grid"><div className="panel"><div className="panel-heading"><div><p className="eyebrow">DOCUMENT INGESTION</p><h3>Upload medical report</h3></div><span className="panel-icon">↥</span></div><p className="panel-description">Text-based PDFs only. The backend stores files locally and runs the existing ingestion → chunking → FAISS indexing pipeline.</p><form className="stack" onSubmit={submitReport}><label className="file-drop"><input type="file" accept="application/pdf,.pdf" onChange={(e) => setSelectedFile(e.target.files?.[0] || null)} required /><span className="file-symbol">↥</span><strong>{selectedFile ? selectedFile.name : 'Choose a PDF file'}</strong><small>{selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB selected` : 'Maximum size configured by the backend · selectable text required'}</small></label><button className="primary" disabled={busy || !selectedFile}>Upload and index <span>→</span></button></form><div className="notice-card"><Icon>ⓘ</Icon><p><strong>Synthetic-only mode is on by default.</strong> To test, upload the existing synthetic sample reports. Real reports are blocked unless you explicitly enable local real-report uploads after a privacy and security review.</p></div></div><div className="panel"><div className="panel-heading"><div><p className="eyebrow">PATIENT KNOWLEDGE BASE</p><h3>Uploaded reports <span className="count-bubble">{reports.length}</span></h3></div></div>{reports.length === 0 ? <Empty title="No uploaded reports yet" detail="Upload a synthetic test PDF to exercise ingestion and indexing." /> : <div className="report-list">{reports.map((report) => <article className="report-row" key={report.report_id || report.id}><div className="pdf-icon">PDF</div><div className="report-copy"><strong>{report.original_filename}</strong><p>{report.page_count || '?'} pages · {formatDate(report.created_at)}</p><small>Index status: <b className={report.index_status === 'indexed' ? 'text-success' : report.index_status === 'failed' ? 'text-danger' : ''}>{report.index_status}</b></small>{report.index_error && <small className="text-danger">{report.index_error}</small>}</div><span className={`status-badge ${report.index_status}`}>{report.index_status}</span></article>)}</div>}</div></section>}

        {tab === 'history' && <section className="page-grid"><div className="panel"><div className="panel-heading"><div><p className="eyebrow">STRUCTURED PATIENT CONTEXT</p><h3>Add history entry</h3></div></div><form className="stack" onSubmit={submitHistory}><label>Category<select value={historyForm.category} onChange={(e) => setHistoryForm({ ...historyForm, category: e.target.value })}><option value="condition">Condition (reported)</option><option value="diagnosis">Diagnosis (reported)</option><option value="medication">Medication</option><option value="allergy">Allergy</option><option value="other">Other</option></select></label><label>Name<input value={historyForm.name} onChange={(e) => setHistoryForm({ ...historyForm, name: e.target.value })} required maxLength={120} /></label><label>Reported on<input type="date" value={historyForm.reported_on} onChange={(e) => setHistoryForm({ ...historyForm, reported_on: e.target.value })} required /></label><label>Details<textarea rows="3" value={historyForm.details} onChange={(e) => setHistoryForm({ ...historyForm, details: e.target.value })} maxLength={500} /></label><button className="primary" disabled={busy}>Save history <span>→</span></button></form></div><div className="panel"><div className="panel-heading"><div><p className="eyebrow">UNVERIFIED ENTRIES</p><h3>Medical history <span className="count-bubble">{history.length}</span></h3></div></div>{history.length === 0 ? <Empty title="No history entries" detail="Caregiver-entered history is kept separate from report evidence." /> : <div className="history-list">{history.map((item) => <article className="history-item" key={item.id}><div className="history-symbol">{item.category === 'medication' ? '＋' : item.category === 'allergy' ? '◇' : '○'}</div><div className="history-copy"><strong>{item.name}</strong><span>{item.category} · {item.verification_status || 'unverified'}</span><p>{item.details || 'No details recorded.'}</p><small>Reported on {String(item.reported_on || '').slice(0, 10)}</small></div></article>)}</div>}</div><div className="panel medication-panel"><div className="panel-heading"><div><p className="eyebrow">MEDICINE-BOX WORKFLOW</p><h3>Add medication record</h3></div></div><form className="form-grid" onSubmit={submitMedication}><label>Name<input value={medicationForm.name} onChange={(e) => setMedicationForm({ ...medicationForm, name: e.target.value })} required /></label><label>Dosage text<input value={medicationForm.dosage_text} onChange={(e) => setMedicationForm({ ...medicationForm, dosage_text: e.target.value })} /></label><label>Schedule text<input value={medicationForm.schedule_text} onChange={(e) => setMedicationForm({ ...medicationForm, schedule_text: e.target.value })} /></label><label>Notes<input value={medicationForm.notes} onChange={(e) => setMedicationForm({ ...medicationForm, notes: e.target.value })} /></label><button className="primary" disabled={busy}>Save medication <span>→</span></button></form><p className="micro-note">The app records caregiver entries and sensor events; it does not prescribe doses or infer that a box opening means a dose was taken.</p><div className="medication-list">{medications.map((med) => <article className="medication-row" key={med.id}><div><strong>{med.name}</strong><p>{med.dosage_text} · {med.schedule_text}</p><small>{med.verification_status || 'unverified'}</small></div><div className="medication-actions"><button className="secondary small-button" onClick={() => recordMedicationEvent(med.id, 'box_opened')}>Box opened</button><button className="secondary small-button" onClick={() => recordMedicationEvent(med.id, 'dose_marked_taken')}>Mark taken</button></div></article>)}</div></div></section>}

        {tab === 'ai' && <section className="panel explanation-panel">
          <div className="panel-heading"><div><p className="eyebrow">RETRIEVAL-AUGMENTED GENERATION</p><h3>Ask Lifora AI</h3></div><span className="mode-tag">Evidence-linked assistant</span></div>
          <p className="panel-description">Ask about recorded symptoms, patient history and uploaded reports. The backend retrieves patient-specific and trusted-reference evidence. The assistant is informational and does not diagnose or replace professional care.</p>
          <form className="explain-form" onSubmit={submitExplanation}><textarea rows="3" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={500} placeholder="Ask a question about the available patient records…" /><button className="primary" disabled={busy}>Generate explanation <span>→</span></button></form>
          {explanation && <div className="explanation-result"><div className="result-top"><span className="result-check">✓</span><div><strong>Explanation generated</strong><small>{explanation.mode?.replaceAll('_', ' ')} · {explanation.model} · {explanation.latency_seconds}s</small></div></div><div className="answer-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{explanation.answer || ""}</ReactMarkdown></div><p className="micro-note">{explanation.validation} This is not a diagnosis.</p><details><summary>View evidence packet and citations</summary><pre>{JSON.stringify(explanation.sources, null, 2)}</pre></details></div>}
        </section>}

        {tab === 'system' && <section className="panel system-panel"><div className="panel-heading"><div><p className="eyebrow">CONFIGURATION / HEALTH CHECK</p><h3>Service status</h3></div><button className="secondary" onClick={loadSystemStatus}>Refresh status ↻</button></div>{!system ? <Empty title="Status not loaded" detail="The system status is fetched from the authenticated API." /> : <div className="system-grid"><div className="system-card"><h4>RAG + LLM</h4><p>Cloud LLM: <b>{system.rag.cloud_llm_enabled ? 'enabled' : 'disabled'}</b></p><p>Groq key configured: <b>{system.rag.groq_key_configured ? 'yes' : 'no'}</b></p><p>Trusted reference index: <b>{system.rag.trusted_reference_index_pointer_exists ? 'present' : 'not built'}</b></p><pre>{JSON.stringify(system.rag.scripts, null, 2)}</pre></div><div className="system-card"><h4>Database & privacy</h4><p>MongoDB: <b>{system.database}</b></p><p>Real PDF uploads: <b>{system.privacy.real_report_uploads_enabled ? 'enabled' : 'synthetic-only'}</b></p><p>Real data to cloud: <b>{system.privacy.real_data_to_cloud_llm_enabled ? 'enabled' : 'blocked'}</b></p><p className="micro-note">{system.privacy.note}</p></div><div className="system-card"><h4>ML integration</h4><pre>{JSON.stringify(system.ml, null, 2)}</pre></div><div className="system-card"><h4>Emergency monitoring</h4><pre>{JSON.stringify(system.emergency_monitoring, null, 2)}</pre></div></div>}</section>}

        <footer className="app-footer"><span>© Lifora prototype</span><span>Patient facts · caregiver observations · trusted references remain distinct</span><span>Not for clinical use</span></footer>
      </main>
    </div>
  );
}
