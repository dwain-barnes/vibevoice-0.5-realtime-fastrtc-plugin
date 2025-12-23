"""
FastRTC Integration Examples for VibeVoice TTS
==============================================

Prerequisites:
    pip install fastrtc

Examples:
    1. tts      - Text-to-Speech UI
    2. echo     - Echo Bot (speak → hear)
    3. chat     - LLM Voice Chat (Custom HTML UI)
    4. server   - WebSocket TTS Server
"""

from vibevoice_fastrtc import VibeVoiceTTS, VibeVoiceOptions

# Initialize TTS globally (lazy loads on first use)
tts = VibeVoiceTTS()
tts_options = VibeVoiceOptions(speaker_name="en-Grace_woman.pt", cfg_scale=1.5)


# =============================================================================
# Example 1: Text-to-Speech UI
# =============================================================================

def run_tts_ui():
    """Simple UI: Type text, click button, hear speech."""
    from fastrtc import Stream
    import gradio as gr
    
    def text_to_speech(text: str):
        if not text.strip():
            return
        for chunk in tts.stream_tts_sync(text, tts_options):
            yield chunk
    
    with gr.Blocks() as demo:
        gr.Markdown("# VibeVoice Text-to-Speech")
        text_input = gr.Textbox(label="Enter text", placeholder="Hello!", lines=3)
        voice_dropdown = gr.Dropdown(
            choices=tts.get_available_voices(),
            value="en-Grace_woman",
            label="Voice"
        )
        stream = Stream(handler=text_to_speech, mode="send", modality="audio")
        speak_btn = gr.Button("🔊 Speak", variant="primary")
        speak_btn.click(fn=lambda t, v: text_to_speech(t), inputs=[text_input, voice_dropdown])
        stream.ui.render()
    
    demo.launch()


# =============================================================================
# Example 2: Echo Bot
# =============================================================================

def run_echo_bot():
    """Speak into mic, hear your words repeated back - Fixed Connection."""
    from fastrtc import Stream, ReplyOnPause, get_stt_model
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import gradio as gr
    import uvicorn

    stt = get_stt_model()
    selected_voice = {"name": "en-Grace_woman"}

    def echo(audio):
        text = stt.stt(audio)
        print(f"You said: {text}")
        if not text.strip():
            return
        voice_options = VibeVoiceOptions(speaker_name=selected_voice["name"], cfg_scale=1.5)
        for chunk in tts.stream_tts_sync(text, voice_options):
            yield chunk

    # 1. Create FastAPI App
    app = FastAPI(title="Echo Bot")

    # 2. Define the Stream
    stream = Stream(
        handler=ReplyOnPause(echo),
        mode="send-receive",
        modality="audio",
    )

    # 3. CRITICAL FIX: Mount stream to root to expose /webrtc/offer
    stream.mount(app)

    # 4. API Routes
    @app.get("/api/voices")
    async def get_voices():
        return JSONResponse(tts.get_available_voices())

    @app.post("/api/voice")
    async def set_voice(request: Request):
        data = await request.json()
        voice = data.get("voice", "en-Grace_woman")
        selected_voice["name"] = voice
        print(f"✓ Voice changed to: {voice}")
        return JSONResponse({"status": "ok", "voice": voice})

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return get_echo_bot_html()

    # 5. Mount Gradio (Optional, for debugging at /gradio)
    with gr.Blocks() as demo:
        stream.ui.render()
    
    app = gr.mount_gradio_app(app, demo, path="/gradio")

    print("=" * 60)
    print("🎤 Echo Bot (Fixed)")
    print("=" * 60)
    print(f"Open browser at: http://localhost:7860")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=7860)


def get_echo_bot_html():
    """Returns the custom HTML UI for Echo Bot."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Echo Bot</title>
    <style>
        :root {
            --accent: #10b981;
            --accent2: #06b6d4;
            --bg: #0f172a;
            --surface: #1e293b;
            --surface2: #334155;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --success: #22c55e;
            --error: #ef4444;
            --warning: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 600px;
            background: var(--surface);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        }
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        .header h1 {
            font-size: 2rem;
            margin-bottom: 8px;
        }
        .header p {
            color: var(--muted);
        }
        .voice-select {
            margin-bottom: 24px;
        }
        .voice-select label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--muted);
        }
        .voice-select select {
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--surface2);
            background: var(--bg);
            color: var(--text);
            font-size: 16px;
            cursor: pointer;
        }
        .voice-select select:focus {
            outline: none;
            border-color: var(--accent);
        }
        .visualizer {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80px;
            gap: 4px;
            margin: 24px 0;
        }
        .bar {
            width: 6px;
            height: 100%;
            background: linear-gradient(to top, var(--accent), var(--accent2));
            border-radius: 3px;
            transform: scaleY(0.1);
            transition: transform 0.05s;
        }
        .status {
            text-align: center;
            margin-bottom: 24px;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--surface2);
            border-radius: 20px;
            font-size: 14px;
        }
        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--muted);
        }
        .dot.connecting { background: var(--warning); animation: pulse 1s infinite; }
        .dot.connected { background: var(--success); animation: pulse 2s infinite; }
        @keyframes pulse { 50% { opacity: 0.5; } }
        .controls {
            display: flex;
            justify-content: center;
            gap: 12px;
        }
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            color: white;
        }
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -10px var(--accent);
        }
        .btn-stop {
            background: var(--error);
            color: white;
        }
        .btn-mute {
            padding: 14px;
            background: var(--surface2);
            color: var(--text);
        }
        .btn-mute.muted { background: var(--error); color: white; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn svg { width: 20px; height: 20px; }
        .spinner {
            width: 20px; height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .tips {
            margin-top: 24px;
            padding: 16px;
            background: rgba(16,185,129,0.1);
            border-left: 3px solid var(--accent);
            border-radius: 8px;
            font-size: 14px;
            color: var(--muted);
        }
        .toast {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            display: none;
            z-index: 1000;
        }
        .toast.success { background: var(--success); color: white; }
        .toast.error { background: var(--error); color: white; }
        .toast.show { display: block; animation: slideIn 0.3s; }
        @keyframes slideIn { from { opacity: 0; transform: translateX(-50%) translateY(-20px); } }
    </style>
</head>
<body>
    <div id="toast" class="toast"></div>
    <div class="container">
        <div class="header">
            <h1>🎤 Echo Bot</h1>
            <p>Speak and hear your words echoed back</p>
        </div>

        <div class="voice-select">
            <label>🔊 Echo Voice</label>
            <select id="voice"></select>
        </div>

        <div class="visualizer" id="viz"></div>

        <div class="status">
            <div class="status-badge">
                <div class="dot" id="dot"></div>
                <span id="status">Ready</span>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" id="startBtn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
                Start
            </button>
            <button class="btn btn-mute" id="muteBtn" style="display:none;">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
            </button>
        </div>

        <div class="tips">
            💡 <strong>Tips:</strong> Click Start and speak. Your words will be transcribed and spoken back in the selected voice.
        </div>
    </div>

    <audio id="audio" autoplay></audio>

    <script>
    const BARS = 32;
    const ICE = {iceServers: [{urls: 'stun:stun.l.google.com:19302'}]};
    
    let pc, dc, ctx, analyser, anim, muted = false, webrtcId = null, workingEndpoint = null;
    
    const $ = id => document.getElementById(id);
    const startBtn = $('startBtn'), muteBtn = $('muteBtn'), voice = $('voice');
    const dot = $('dot'), status = $('status'), viz = $('viz'), audio = $('audio'), toast = $('toast');
    
    for(let i = 0; i < BARS; i++) {
        const b = document.createElement('div');
        b.className = 'bar';
        viz.appendChild(b);
    }
    const bars = viz.querySelectorAll('.bar');
    
    fetch('/api/voices').then(r => r.json()).then(voices => {
        voice.innerHTML = voices.map(v => 
            `<option value="${v}">${v.replace('.pt','').replace(/_/g,' ')}</option>`
        ).join('');
    }).catch(() => {
        voice.innerHTML = '<option value="en-Grace_woman">en-Grace woman</option>';
    });
    
    voice.onchange = () => {
        fetch('/api/voice', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({voice: voice.value})
        });
        showToast('Voice: ' + voice.value.replace('.pt','').replace(/_/g,' '), 'success');
    };
    
    startBtn.onclick = () => pc && pc.connectionState === 'connected' ? stop() : start();
    muteBtn.onclick = toggleMute;
    
    async function start() {
        setStatus('connecting', 'Connecting...');
        startBtn.disabled = true;
        startBtn.innerHTML = '<div class="spinner"></div> Connecting...';
        webrtcId = Math.random().toString(36).substring(2, 9);
        
        try {
            pc = new RTCPeerConnection(ICE);
            const stream = await navigator.mediaDevices.getUserMedia({audio: true});
            stream.getTracks().forEach(t => pc.addTrack(t, stream));
            
            ctx = new AudioContext();
            analyser = ctx.createAnalyser();
            analyser.fftSize = 64;
            ctx.createMediaStreamSource(stream).connect(analyser);
            
            dc = pc.createDataChannel('text');
            
            pc.ontrack = e => {
                audio.srcObject = e.streams[0];
                const a2 = ctx.createAnalyser();
                a2.fftSize = 64;
                ctx.createMediaStreamSource(e.streams[0]).connect(a2);
                analyser = a2;
            };
            
            pc.onconnectionstatechange = () => {
                console.log('Connection state:', pc.connectionState);
                if(pc.connectionState === 'connected') {
                    setStatus('connected', 'Listening...');
                    startBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg> Stop';
                    startBtn.className = 'btn btn-stop';
                    startBtn.disabled = false;
                    muteBtn.style.display = 'flex';
                    animate();
                } else if(['disconnected','failed','closed'].includes(pc.connectionState)) {
                    stop();
                }
            };
            
            pc.onicecandidate = async ({candidate}) => {
                if(candidate && workingEndpoint) {
                    await fetch(workingEndpoint, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({type: 'ice-candidate', candidate: candidate.toJSON(), webrtc_id: webrtcId})
                    });
                }
            };
            
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            const endpoints = ['/webrtc/offer', '/gradio/webrtc/offer'];
            let answer = null;
            
            for(const endpoint of endpoints) {
                try {
                    console.log('Trying endpoint:', endpoint);
                    const res = await fetch(endpoint, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            type: offer.type,
                            sdp: offer.sdp,
                            webrtc_id: webrtcId
                        })
                    });
                    
                    const data = await res.json();
                    console.log('Response from', endpoint, ':', data);
                    
                    if(data.sdp && data.type) {
                        answer = data;
                        workingEndpoint = endpoint;
                        break;
                    } else if(data.status !== 'failed') {
                        answer = {type: 'answer', sdp: data.sdp || data};
                        workingEndpoint = endpoint;
                        break;
                    }
                } catch(e) {
                    console.log('Endpoint', endpoint, 'failed:', e.message);
                }
            }
            
            if(!answer || !answer.sdp) {
                throw new Error('No WebRTC endpoint responded correctly');
            }
            
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
            
        } catch(e) {
            console.error('Connection error:', e);
            showToast(e.message, 'error');
            stop();
        }
    }
    
    function stop() {
        if(anim) cancelAnimationFrame(anim);
        if(pc) { pc.getSenders().forEach(s => s.track?.stop()); pc.close(); }
        if(dc) dc.close();
        if(ctx) ctx.close();
        pc = dc = ctx = analyser = null;
        webrtcId = null;
        workingEndpoint = null;
        audio.srcObject = null;
        muted = false;
        
        setStatus('ready', 'Ready');
        startBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
        </svg> Start`;
        startBtn.className = 'btn btn-primary';
        startBtn.disabled = false;
        muteBtn.style.display = 'none';
        muteBtn.classList.remove('muted');
        bars.forEach(b => b.style.transform = 'scaleY(0.1)');
    }
    
    function toggleMute() {
        muted = !muted;
        pc?.getSenders().forEach(s => { if(s.track?.kind === 'audio') s.track.enabled = !muted; });
        muteBtn.classList.toggle('muted', muted);
        status.textContent = muted ? 'Muted' : 'Listening...';
    }
    
    function animate() {
        if(!pc || pc.connectionState !== 'connected') return;
        if(analyser) {
            const data = new Uint8Array(BARS);
            analyser.getByteFrequencyData(data);
            bars.forEach((b,i) => b.style.transform = `scaleY(${Math.max(0.1, data[i]/255 * 1.5)})`);
        }
        anim = requestAnimationFrame(animate);
    }
    
    function setStatus(state, text) {
        dot.className = 'dot' + (state !== 'ready' ? ' ' + state : '');
        status.textContent = text;
    }
    
    function showToast(msg, type) {
        toast.textContent = msg;
        toast.className = `toast ${type} show`;
        setTimeout(() => toast.classList.remove('show'), 3000);
    }
    </script>
</body>
</html>'''


# =============================================================================
# Example 3: LLM Voice Chat with Custom HTML UI
# =============================================================================

def run_llm_chat():
    """Voice chat with an LLM - custom HTML frontend, Gradio/FastRTC backend."""
    from fastrtc import Stream, ReplyOnPause, get_stt_model
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import gradio as gr
    import requests
    import traceback
    import uvicorn

    stt = get_stt_model()

    system_instruction = (
        "You're in a speech conversation with a user. Their text is being transcribed using "
        "speech-to-text. Your responses will be spoken out loud, so don't worry about formatting "
        "and don't use unpronouncable characters like emojis and *. "
        "Everything is pronounced literally, so things like '(chuckles)' won't work. "
        "Write as a human would speak. Respond to the user's text as if you were having a casual "
        "conversation with them. Respond in the language the user is speaking."
    )

    messages = [{"role": "system", "content": system_instruction}]
    selected_voice = {"name": "en-Grace_woman"}

    def get_llm_response(chat_history) -> str:
        try:
            # Using standard OpenAI format (compatible with LM Studio, Ollama, etc)
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json={"messages": chat_history, "max_tokens": 300, "temperature": 0.7},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return "I'm having trouble thinking."
        except Exception:
            return "I cannot connect to the brain."

    def voice_chat(audio):
        try:
            user_text = stt.stt(audio)
            if not user_text.strip():
                return
            print(f"\n🎤 User: {user_text}")

            messages.append({"role": "user", "content": user_text})
            response_text = get_llm_response(messages)
            print(f"🤖 Assistant: {response_text}")
            messages.append({"role": "assistant", "content": response_text})

            if response_text:
                voice_options = VibeVoiceOptions(speaker_name=selected_voice["name"], cfg_scale=1.5)
                for chunk in tts.stream_tts_sync(response_text, voice_options):
                    yield chunk
        except Exception as e:
            print(f"Pipeline Error: {e}")
            traceback.print_exc()

    # 1. Create FastAPI app
    app = FastAPI(title="AI Voice Chat")

    # 2. Define the Stream
    stream = Stream(
        handler=ReplyOnPause(voice_chat),
        mode="send-receive",
        modality="audio",
    )

    # 3. CRITICAL FIX: Mount the stream endpoints to the root of FastAPI
    # This enables "POST /webrtc/offer" to work
    stream.mount(app)

    # Custom API routes
    @app.get("/api/voices")
    async def get_voices():
        return JSONResponse(tts.get_available_voices())

    @app.post("/api/voice")
    async def set_voice(request: Request):
        data = await request.json()
        voice = data.get("voice", "en-Grace_woman")
        selected_voice["name"] = voice
        print(f"✓ Voice changed to: {voice}")
        return JSONResponse({"status": "ok", "voice": voice})

    @app.get("/", response_class=HTMLResponse)
    async def custom_index():
        return get_voice_chat_html()

    # Optional: Keep Gradio UI at /gradio if you want to debug visually
    with gr.Blocks() as demo:
        stream.ui.render()
    app = gr.mount_gradio_app(app, demo, path="/gradio")

    print("=" * 60)
    print("🤖 AI Voice Chat")
    print("=" * 60)
    print(f"Open browser at: http://localhost:7860")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=7860)


def get_voice_chat_html():
    """Returns the custom HTML UI."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Chat</title>
    <style>
        :root {
            --accent: #6366f1;
            --bg: #0f172a;
            --surface: #1e293b;
            --surface2: #334155;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --success: #22c55e;
            --error: #ef4444;
            --warning: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 600px;
            background: var(--surface);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        }
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        .header h1 {
            font-size: 2rem;
            margin-bottom: 8px;
        }
        .header p {
            color: var(--muted);
        }
        .voice-select {
            margin-bottom: 24px;
        }
        .voice-select label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--muted);
        }
        .voice-select select {
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--surface2);
            background: var(--bg);
            color: var(--text);
            font-size: 16px;
            cursor: pointer;
        }
        .voice-select select:focus {
            outline: none;
            border-color: var(--accent);
        }
        .visualizer {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80px;
            gap: 4px;
            margin: 24px 0;
        }
        .bar {
            width: 6px;
            height: 100%;
            background: linear-gradient(to top, var(--accent), #a855f7);
            border-radius: 3px;
            transform: scaleY(0.1);
            transition: transform 0.05s;
        }
        .status {
            text-align: center;
            margin-bottom: 24px;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--surface2);
            border-radius: 20px;
            font-size: 14px;
        }
        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--muted);
        }
        .dot.connecting { background: var(--warning); animation: pulse 1s infinite; }
        .dot.connected { background: var(--success); animation: pulse 2s infinite; }
        .dot.error { background: var(--error); }
        @keyframes pulse { 50% { opacity: 0.5; } }
        .controls {
            display: flex;
            justify-content: center;
            gap: 12px;
        }
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent), #a855f7);
            color: white;
        }
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -10px var(--accent);
        }
        .btn-stop {
            background: var(--error);
            color: white;
        }
        .btn-mute {
            padding: 14px;
            background: var(--surface2);
            color: var(--text);
        }
        .btn-mute.muted { background: var(--error); color: white; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn svg { width: 20px; height: 20px; }
        .spinner {
            width: 20px; height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .tips {
            margin-top: 24px;
            padding: 16px;
            background: rgba(99,102,241,0.1);
            border-left: 3px solid var(--accent);
            border-radius: 8px;
            font-size: 14px;
            color: var(--muted);
        }
        .toast {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            display: none;
            z-index: 1000;
        }
        .toast.success { background: var(--success); color: white; }
        .toast.error { background: var(--error); color: white; }
        .toast.show { display: block; animation: slideIn 0.3s; }
        @keyframes slideIn { from { opacity: 0; transform: translateX(-50%) translateY(-20px); } }
    </style>
</head>
<body>
    <div id="toast" class="toast"></div>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Voice Chat</h1>
            <p>Speak to the AI and hear its response</p>
        </div>

        <div class="voice-select">
            <label>🔊 Assistant Voice</label>
            <select id="voice"></select>
        </div>

        <div class="visualizer" id="viz"></div>

        <div class="status">
            <div class="status-badge">
                <div class="dot" id="dot"></div>
                <span id="status">Ready</span>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" id="startBtn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
                Start
            </button>
            <button class="btn btn-mute" id="muteBtn" style="display:none;">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
            </button>
        </div>

        <div class="tips">
            💡 <strong>Tips:</strong> Click Start, then speak clearly. Wait for the AI response before speaking again.
        </div>
    </div>

    <audio id="audio" autoplay></audio>

    <script>
    const BARS = 32;
    const ICE = {iceServers: [{urls: 'stun:stun.l.google.com:19302'}]};
    
    let pc, dc, ctx, analyser, anim, muted = false, webrtcId = null;
    
    const $ = id => document.getElementById(id);
    const startBtn = $('startBtn'), muteBtn = $('muteBtn'), voice = $('voice');
    const dot = $('dot'), status = $('status'), viz = $('viz'), audio = $('audio'), toast = $('toast');
    
    // Create bars
    for(let i = 0; i < BARS; i++) {
        const b = document.createElement('div');
        b.className = 'bar';
        viz.appendChild(b);
    }
    const bars = viz.querySelectorAll('.bar');
    
    // Load voices
    fetch('/api/voices').then(r => r.json()).then(voices => {
        voice.innerHTML = voices.map(v => 
            `<option value="${v}">${v.replace('.pt','').replace(/_/g,' ')}</option>`
        ).join('');
    }).catch(() => {
        voice.innerHTML = '<option value="en-Grace_woman">en-Grace woman</option>';
    });
    
    voice.onchange = () => {
        fetch('/api/voice', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({voice: voice.value})
        });
        showToast('Voice: ' + voice.value.replace('.pt','').replace(/_/g,' '), 'success');
    };
    
    startBtn.onclick = () => pc && pc.connectionState === 'connected' ? stop() : start();
    muteBtn.onclick = toggleMute;
    
    async function start() {
        setStatus('connecting', 'Connecting...');
        startBtn.disabled = true;
        startBtn.innerHTML = '<div class="spinner"></div> Connecting...';
        webrtcId = Math.random().toString(36).substring(2, 9);
        
        try {
            pc = new RTCPeerConnection(ICE);
            const stream = await navigator.mediaDevices.getUserMedia({audio: true});
            stream.getTracks().forEach(t => pc.addTrack(t, stream));
            
            ctx = new AudioContext();
            analyser = ctx.createAnalyser();
            analyser.fftSize = 64;
            ctx.createMediaStreamSource(stream).connect(analyser);
            
            dc = pc.createDataChannel('text');
            
            pc.ontrack = e => {
                audio.srcObject = e.streams[0];
                const a2 = ctx.createAnalyser();
                a2.fftSize = 64;
                ctx.createMediaStreamSource(e.streams[0]).connect(a2);
                analyser = a2;
            };
            
            pc.onconnectionstatechange = () => {
                console.log('Connection state:', pc.connectionState);
                if(pc.connectionState === 'connected') {
                    setStatus('connected', 'Listening...');
                    startBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg> Stop';
                    startBtn.className = 'btn btn-stop';
                    startBtn.disabled = false;
                    muteBtn.style.display = 'flex';
                    animate();
                } else if(['disconnected','failed','closed'].includes(pc.connectionState)) {
                    stop();
                }
            };
            
            // Store the working endpoint
            let workingEndpoint = null;
            
            pc.onicecandidate = async ({candidate}) => {
                if(candidate && workingEndpoint) {
                    await fetch(workingEndpoint, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({type: 'ice-candidate', candidate: candidate.toJSON(), webrtc_id: webrtcId})
                    });
                }
            };
            
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            // Try different endpoints to find FastRTC
            const endpoints = ['/webrtc/offer', '/gradio/webrtc/offer'];
            let answer = null;
            
            for(const endpoint of endpoints) {
                try {
                    console.log('Trying endpoint:', endpoint);
                    const res = await fetch(endpoint, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            type: offer.type,
                            sdp: offer.sdp,
                            webrtc_id: webrtcId
                        })
                    });
                    
                    const data = await res.json();
                    console.log('Response from', endpoint, ':', data);
                    
                    if(data.sdp && data.type) {
                        answer = data;
                        workingEndpoint = endpoint;
                        break;
                    } else if(data.status !== 'failed') {
                        // Maybe it's just the SDP without explicit type
                        answer = {type: 'answer', sdp: data.sdp || data};
                        workingEndpoint = endpoint;
                        break;
                    }
                } catch(e) {
                    console.log('Endpoint', endpoint, 'failed:', e.message);
                }
            }
            
            if(!answer || !answer.sdp) {
                throw new Error('No WebRTC endpoint responded correctly');
            }
            
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
            
        } catch(e) {
            console.error('Connection error:', e);
            showToast(e.message, 'error');
            stop();
        }
    }
    
    function stop() {
        if(anim) cancelAnimationFrame(anim);
        if(pc) { pc.getSenders().forEach(s => s.track?.stop()); pc.close(); }
        if(dc) dc.close();
        if(ctx) ctx.close();
        pc = dc = ctx = analyser = null;
        webrtcId = null;
        audio.srcObject = null;
        muted = false;
        
        setStatus('ready', 'Ready');
        startBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
        </svg> Start`;
        startBtn.className = 'btn btn-primary';
        startBtn.disabled = false;
        muteBtn.style.display = 'none';
        muteBtn.classList.remove('muted');
        bars.forEach(b => b.style.transform = 'scaleY(0.1)');
    }
    
    function toggleMute() {
        muted = !muted;
        pc?.getSenders().forEach(s => { if(s.track?.kind === 'audio') s.track.enabled = !muted; });
        muteBtn.classList.toggle('muted', muted);
        status.textContent = muted ? 'Muted' : 'Listening...';
    }
    
    function animate() {
        if(!pc || pc.connectionState !== 'connected') return;
        if(analyser) {
            const data = new Uint8Array(BARS);
            analyser.getByteFrequencyData(data);
            bars.forEach((b,i) => b.style.transform = `scaleY(${Math.max(0.1, data[i]/255 * 1.5)})`);
        }
        anim = requestAnimationFrame(animate);
    }
    
    function setStatus(state, text) {
        dot.className = 'dot' + (state !== 'ready' ? ' ' + state : '');
        status.textContent = text;
    }
    
    function showToast(msg, type) {
        toast.textContent = msg;
        toast.className = `toast ${type} show`;
        setTimeout(() => toast.classList.remove('show'), 3000);
    }
    </script>
</body>
</html>'''


# =============================================================================
# Example 4: WebSocket Server
# =============================================================================

def run_websocket_server(host: str = "0.0.0.0", port: int = 8765):
    """WebSocket server for TTS."""
    import asyncio
    import json
    import struct
    
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets")
        return
    
    async def handle_client(websocket):
        print(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    text = data.get("text", "")
                    voice = data.get("voice", "en-Grace_woman")
                except json.JSONDecodeError:
                    text = message
                    voice = "en-Grace_woman"
                
                if not text.strip():
                    continue
                
                print(f"Generating: '{text}' with voice '{voice}'")
                options = VibeVoiceOptions(speaker_name=voice)
                
                for sample_rate, chunk in tts.stream_tts_sync(text, options):
                    header = struct.pack("<I", sample_rate)
                    await websocket.send(header + chunk.tobytes())
                
                await websocket.send(b"END")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print(f"Client disconnected: {websocket.remote_address}")
    
    async def main():
        print(f"Starting WebSocket TTS server on ws://{host}:{port}")
        async with websockets.serve(handle_client, host, port):
            await asyncio.Future()
    
    asyncio.run(main())


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("VibeVoice FastRTC Examples")
    print("=" * 60)
    print()
    print("Available examples:")
    print("  1. tts      - Text-to-Speech UI")
    print("  2. echo     - Echo Bot (speak → hear)")
    print("  3. chat     - LLM Voice Chat (Custom HTML)")
    print("  4. server   - WebSocket TTS Server")
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python fastrtc_examples.py <example>")
        print("Example: python fastrtc_examples.py chat")
        sys.exit(1)
    
    example = sys.argv[1].lower()
    
    if example in ("1", "tts"):
        run_tts_ui()
    elif example in ("2", "echo"):
        run_echo_bot()
    elif example in ("3", "chat"):
        run_llm_chat()
    elif example in ("4", "server"):
        run_websocket_server()
    else:
        print(f"Unknown example: {example}")
        sys.exit(1)
