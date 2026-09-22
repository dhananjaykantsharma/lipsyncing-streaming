# MuseTalk Lip-Sync Avatar Pipeline (Modal A100) — Project Summary

> Ye document simple Hinglish mein likha hai. Jahan technical baat aayi hai, wahan roz-marra ka example diya hai.
> Last update: 2026-09-21. Status: **end-to-end kaam kar raha hai. Asli Modal server pe 60 sec ka clip awaaz ke saath fixed 25 fps pe chala: 1500/1500 frames, 0 skipped, 0 late.** Server warm hone pe ~45-48 fps banata hai (real-time ka ~2x). Baaki: cold start (~40 s), pehle frame ki delay (audio upload + US-West ka network), aur deploy + auth.

---

## 1. Hum kya bana rahe hain (goal)

Ek interview application hai. Usme ek **AI interviewer ka chehra (avatar)** dikhana hai. Jab interviewer kuch bole (audio), to avatar ke **hoth us audio ke hisaab se hilne chahiye** (lip-sync), aur ye video candidate ki screen pe **live stream** hona chahiye.

**Example:** Interviewer ka sawal hai *"Apne baare mein bataiye"*. Ye audio hum system ko dete hain. System ek fixed avatar video (jisme ek insaan camera ki taraf dekh raha hai) leta hai, aur uske hoth is sentence ke saath sync karke naya video banata hai. Candidate ko lagta hai ki avatar sach mein bol raha hai.

**Tools:**
- **MuseTalk** — open-source AI model jo lip-sync karta hai.
- **Modal.com** — cloud jahan hum GPU (A100 40GB) kiraye pe lete hain (jitni der chalega, utne ka paisa).
- **WebSocket** — ek "phone call jaisi line" jo client aur server ke beech khuli rehti hai, taaki data dono taraf turant aa-ja sake.

---

## 2. Pura flow (start se end tak)

### 2.1 Ek baar ka setup (offline, sirf ek baar karna hota hai)

```
[Internet: HuggingFace]                       [Tumhari avatar video: sample.mp4]
        │                                                  │
        ▼                                                  ▼
 download_weights()                                 preprocess_avatar()
 (6 AI models download)                             (chehra dhundo, crop karo,
        │                                            AI-format mein convert karo)
        └──────────────┐                          ┌───────┘
                       ▼                          ▼
              ┌──────────────────────────────────────┐
              │  Modal Volume "musetalk-data"         │
              │  (ek permanent hard-disk jaisa)       │
              │   /models  -> saare AI models         │
              │   /avatars -> avatar ka ready data    │
              └──────────────────────────────────────┘
```

**Example:** Ye waisa hi hai jaise restaurant khulne se pehle **saman laake fridge mein rakh dena** (models) aur **sabziyaan kaat ke rakh dena** (avatar preprocessing). Jab order aayega, tab ye sab dobara nahi karna padega.

### 2.2 Har request ka live flow (jab candidate ke liye avatar bolta hai)

```
 Interview App                    Modal (US West)    ── A100 GPU container ──────────────┐
 (candidate ki screen)                                                                  │
      │                                                                                 │
      │ 1) Audio bhejo (WebSocket)                                                      │
      │ ───────────────────────────────►  2) Audio ko "AI ki samajh wali form"          │
      │                                       mein badlo (Whisper)                      │
      │                                                                                 │
      │                                    3) GPU: har frame ke liye hoth ki shape      │
      │                                       banao (UNet + VAE)   ~11 ms/frame         │
      │                                                                                 │
      │                                    4) Hoth wale hisse ko avatar ke asli frame   │
      │                                       mein "chipkao" (fast blend)               │
      │                                                                                 │
      │                                    5) Frame ko H.264 mein compress karo         │
      │                                       ~11 ms/frame, ~4 KB/frame                 │
      │ 6) Frames wapas aate hain                                                       │
      │ ◄───────────────────────────────                                                │
      │                                                                                 │
      ▼                                    └──────────────────────────────────────────┘
 7) Frames 25 fps pe dikhao + audio saath bajao (audio = master clock)
```

**Example (assembly line):** Audio = kachcha maal. Step 2-5 = factory ki machines. Step 6 = truck jo maal candidate tak pahunchata hai. Factory tez ho sakti hai, lekin agar **truck (network) dheere hai to customer tak maal dheere hi pahunchega.** Shuru mein hamare case mein yahi ho raha tha (JPEG frames bahut bhaari the). H.264 se ye truck wali dikkat hal ho gayi (Section 5 dekho).

### 2.3 Do tareeke jinse video wapas bhej sakte hain

| Tareeka | Kaise kaam karta hai | Example |
|---|---|---|
| **Full-video mode** (`/ws`) | Poora video ban jaye, tab ek saath bhejo | Match khatam hone ke baad **highlights** dikhana |
| **Frame-streaming mode** (`/ws-stream`) | Har frame banta jaye, **H.264 mein compress hoke** turant bhejte jao | **Live commentary** — jo ho raha hai wahi turant dikh raha hai |

Interview ke liye streaming chahiye, kyunki candidate ko 3 minute wait nahi karwa sakte.

### 2.4 Client video ko 25 fps pe kaise chalata hai (audio ke saath)

Server frames **playback se tez** bhejta hai (~45 fps). Agar client unhe "jaise hi aaye waise hi" dikha de, to video 2x speed se bhagega. Isliye client ye karta hai:

1. Frames ko chhote H.264 packets (~4 KB) ke roop mein **queue** mein jama karta hai.
2. **0.5 sec** ka video jama hone pe audio aur video **ek saath** shuru karta hai (ye jitter cushion hai).
3. **Audio master clock hai.** Har ~2 ms pe dekhta hai ki audio abhi kitne second pe hai, aur video ka wahi frame dikhata hai (position × 25).
4. Agar agla frame abhi nahi aaya to **last frame hold** karta hai (video ruk jata hai, audio nahi rukta). Agar peeche reh gaya to beech ke frames **skip** karke audio ke saath wapas mil jata hai. Skip karte waqt bhi packets decode hote hain, kyunki H.264 mein har frame pichle frame pe depend karta hai.

**Example:** TV pe cricket ki live commentary aur video. Commentary (audio) rukti nahi, video ko uske saath chalna padta hai. Video thoda atke to wo beech ke frames chhod ke commentary ke saath wapas aa jata hai.

---

## 3. Ab tak jo bana (files)

| File | Kaam |
|---|---|
| `image.py` | Cloud container ka "recipe" — kaunse software/libraries install hongi (torch, MuseTalk, mmcv, ffmpeg...). |
| `volume.py` | Permanent storage (`musetalk-data`) ka naam aur folder paths. |
| `preprocess.py` | Ek baar chalne wale kaam: `download_weights()` (6 models) aur `preprocess_avatar()` (avatar ka data ready karna). |
| `server.py` | **Asli service.** GPU class `MuseTalkInference`: models ek baar load karta hai, `/ws` (poora mp4) aur `/ws-stream` (H.264, parallel pipeline: GPU → blend threads → encoder) routes deta hai, aur har stage ki timing measure karta hai. |
| `blendutil.py` | `fast_blend`: MuseTalk ke blend ka tez version (sirf chehre ke aas-paas ka crop chhoota hai). Original se **bit-exact** verify kiya, 8x tez. |
| `h264util.py` | H.264 encoder/decoder helper (PyAV + libx264), server aur test client dono use karte hain. Har frame = ek complete packet, isliye browser (WebCodecs) mein bhi seedha chalega. |
| `tools/ws_h264_view.py` | **Player:** audio bhejta hai, H.264 frames ko **fixed 25 fps** pe dikhata hai aur **audio saath bajata hai** (audio hi master clock hai). Latency/pacing numbers bhi print karta hai. |
| `inspect_musetalk.py` | Chhote helper scripts jo MuseTalk ke repo ke andar jhaank ke (README, requirements, scripts, blending code) sahi API dhoondhne ke kaam aaye. `modal run inspect_musetalk.py::fetch_audio` se sample audio (`test_audio.wav`) local pe aa jata hai. |
| `test.py` | Sabse pehla GPU sanity test. |
| `sample.mp4` | Avatar video (isme awaaz nahi hai). |
| `test_audio.wav`, `test_output.mp4` | Test audio (60 sec) aur uska bana hua output. |

> Note: player `tools/` mein hai. Kuch testing scripts abhi temporary folder mein hain, project mein save nahi hain: network bandwidth diagnostic (`bw_server.py`, `bw_client.py`), blend ka equivalence test, aur local mock server (jisse player bina GPU kharch ke test hua).

---

## 4. Ab tak kya-kya kiya (step by step)

### Phase 0 — Environment banana
- Modal account + CLI setup, `modal.Image` define ki (Python 3.10, ffmpeg, MuseTalk repo clone).
- **Problems jo aayi aur unke fix** (ye MuseTalk ke repo ki purani/mismatch dependencies ki wajah se aayi, hamari galti nahi):

| Problem | Wajah | Fix |
|---|---|---|
| `No module named 'image'` aur 17 containers crash-loop | Modal sirf entry file bhejta hai, dusri local file nahi | `add_local_python_source(...)` (ab: image, volume, h264util, blendutil) |
| `torch_vision` install fail | Package ka naam `torchvision` hai | Naam theek kiya |
| NumPy 2.x vs torch warning | Naya numpy purane torch ke saath compatible nahi | `numpy==1.23.5` |
| `torch.xpu` error | MuseTalk ka `diffusers==0.30.2` naye torch (2.4+) maangta hai | `diffusers==0.27.2` |
| `cached_download` import error | Purana diffusers ko purana huggingface_hub chahiye | `huggingface_hub==0.23.4` |
| `mmcv` compile fail | Hamare torch/CUDA combo ka ready-made mmcv nahi tha | MuseTalk README ka exact combo: `torch 2.0.1 + cu118` |
| `mmpose` missing | Face landmark ke liye OpenMMLab chahiye | `openmim` se `mmengine, mmcv 2.0.1, mmdet 3.1.0, mmpose 1.1.0` |
| `gdown --id` error | Naye gdown mein flag hata diya | ID seedha argument bana di |
| `av==14.4.0` install fail | Us version ka wheel nahi tha, source se build ki koshish hui | `av==17.1.0` (server aur local dono mein) |

### Phase 1 — Data taiyaar karna (complete ✅)
- 6 models Volume mein download: MuseTalk (v1.0 + v1.5), SD-VAE, Whisper-tiny, DWPose, SyncNet, face-parse-bisent.
- `sample.mp4` Volume mein upload (`avatars/avatar.mp4`).
- Avatar preprocessing chala (~1.5 min, 384 frames): chehra detect, crop, latents, masks — sab Volume mein cache. **Ye ab dobara nahi karna padega.**
- Trick: MuseTalk ki apni `Avatar` class reuse ki. Us class mein kuch cheezein "global variables" se aati hain, isliye unko manually inject karna pada.

### Phase 2 — Live service (complete ✅)
- `server.py` mein GPU class: `@modal.enter()` mein models + cached avatar ek baar load.
- `/ws` (poora video), `/ws-stream` (frame-by-frame) dono chal rahe hain.
- Bug fix: GPU ka lamba kaam seedha WebSocket handler mein chalane se connection timeout ho raha tha → **alag thread** mein chalaya.
- Batch size 8 → 20 kiya (MuseTalk ka apna default).
- Streaming ke liye `Avatar.inference()` ko generator mein rewrite kiya (har frame ready hote hi bhejta hai, disk pe PNG nahi likhta). Ye pehla version JPEG frames bhejta tha, baad mein Phase 4 mein H.264 pipeline se replace hua.
- Per-stage timing instrumentation add ki (Section 5 ke numbers wahi se aaye).

### Phase 3 — Latency ki jaanch (complete ✅)
- Streaming mein pehla frame jaldi aaya, lekin poora clip bahut slow.
- Measure kiya to pata chala: **GPU aur server tez hain, asli rukavat network hai.**
- Modal ka server US West (`westus3`) mein nikla, tumhare machine se ~350 ms RTT.
- Ek alag CPU-only diagnostic (`bw_server.py`) se pakka hua ki **bina GPU ke, sirf random bytes bhejne pe bhi** ek connection ~2 Mbps hi deta hai.
- Server abhi **band hai**, kuch deploy nahi kiya (`modal deploy` abhi nahi hua).

### Phase 4 — H.264 + parallel pipeline (kiya ✅)
- JPEG frames hataye, **H.264 (libx264, zerolatency, ~1 keyframe/sec)** lagaya. Har frame ka ek packet, decoder mein koi delay nahi (local test mein confirm).
- Pipeline assembly-line bana: GPU thread → blend ke worker threads (pehle 4, Phase 5 ke baad 2) → encoder thread. Frames order mein hi nikalte hain.
- Container ko `cpu=8` diya. Client (`tools/ws_h264_view.py`) H.264 decode karta hai.
- Natija Section 5.8 mein.

### Phase 5 — Blend ko GIL se bahar nikalna (kiya ✅)
- Original blend har frame pe poora 1280×720 image PIL mein convert karke wapas convert karta tha, jabki badalta sirf chehre ke aas-paas ka hissa hai. `fast_blend` (`blendutil.py`) sirf us crop pe kaam karta hai.
- **Verify kiya:** asli frames/masks pe, saare 384 geometry pe, aur 600 random geometry (358 frame ke edge/bahar wale) pe original se **ek bhi pixel alag nahi**.
- Local test: 12.8 → 1.56 ms/frame (8.2x). Original blend 4 threads pe sirf 1.17x scale hota tha (GIL ka saboot), naya 1.86x.
- Blend threads 4 → 2. Natija Section 5.9 mein.

### Phase 6 — 25 fps player + audio (kiya ✅)
- `tools/ws_h264_view.py` ko player bana diya: frames fixed **25 fps** pe dikhata hai aur **audio saath bajata hai** (Section 2.4).
- Pehle **local mock server** pe test kiya (bina GPU kharch ke): tez delivery pe 25.00 fps, 0 skipped. 1 sec ke stall pe 0.6 s hold hua aur 14 frames skip karke audio ke saath sync wapas aaya. Audio decode aur callback bhi unit-test kiye (bina awaaz ke).
- Ek bug mila aur fix hua: pehli `imshow` window banane mein ~0.4 s leti thi aur 10 frames skip ho jate the. Ab window playback shuru hone se pehle bana lete hain.
- Phir **asli Modal server pe** chalaya: 60 sec ka clip, awaaz ke saath, poora clean (Section 5.10).

---

## 5. Latency (speed) ki poori jaankari

### 5.1 Pehle 3 shabd simple mein

| Shabd | Matlab | Example |
|---|---|---|
| **Cold start** | Machine ka pehli baar chalu hona | Subah dukaan kholna: shutter uthao, machine chalao, saman lagao. Pehle customer ko wait karna padta hai. |
| **RTT (round-trip time)** | Sawal bhejne se jawab aane tak ka time | India se US ko voice note bhejo aur reply aaye — har baar aana-jaana ka delay. |
| **Bandwidth** | Network ki "pipe" kitni mota hai | Pipe patli ho to paani (data) dheere aayega, chahe tanki (server) kitni bhi bhari ho. |
| **FPS** | Ek second mein kitne frames | Real-time avatar ke liye **25 frames/second** chahiye, matlab har frame ke liye sirf **40 millisecond**. |

### 5.2 Server ke andar ka time (shuruaati JPEG + serial version, ab purana; naye numbers 5.9 mein)

| Stage | Time/frame | Matlab |
|---|---|---|
| GPU (hoth ki shape banana) | **~11 ms** | Bahut fast, A100 ko koi dikkat nahi |
| Blend (hoth ko avatar pe chipkana) | ~25 ms | CPU ka kaam |
| JPEG banana | ~23 ms | CPU ka kaam |
| **Total** | **~59 ms** | ~**16.5 fps** — 25fps ke liye thoda slow, par bottleneck ye nahi hai |

Server ne 250 frames (10 sec audio) sirf **15.2 second** mein tayar kar diye.

### 5.3 Cold vs warm

| Cheez | Cold (pehli request) | Warm (agli requests) |
|---|---|---|
| Container + models load | ~90 s | 0 s |
| Audio ko process karna | **23 s** (one-time warm-up) | 0.02 s |
| Pehla GPU batch | 4.7 s | 0.2 s |
| **Pehla frame aane mein (streaming)** | **~31 s** | **~2.3 s** (server ka hissa sirf 0.28 s) |

**Example:** Warm = dukaan khuli hai, customer aate hi chai mil gayi. Cold = dukaan abhi khul rahi hai.
Cold start ka fix: `min_containers=1` (ek container hamesha chalu) aur startup pe ek dummy request chala ke warm-up. Ye paisa lagata hai, kyunki GPU idle bhi chalu rehta hai.

### 5.4 Har mode ke numbers (60 sec ke audio pe, 1500 frames)

| Mode | Pehla frame/result | Poora hone mein | Effective speed |
|---|---|---|---|
| `modal run` (poora video, cold) | — | **3m 47s** (isme ~138s generation + ~89s load) | ~11 fps generation |
| WebSocket `/ws` poora video (cold-ish) | 194 s baad sab kuch ek saath | 194 s | — |
| WebSocket `/ws` poora video (warm) | 164 s baad sab kuch ek saath | 164 s | — |
| WebSocket `/ws-stream` frame-by-frame | **32.7 s** | **533 s** | **2.8 fps** |

**Seekh:** Streaming se **pehla frame 5-6 guna jaldi** aaya (164 s → 33 s), lekin poora clip zyada der mein pahuncha, kyunki network dheere frame utha raha tha.

### 5.5 Network ki jaanch (bottleneck yahan hai)

| Measure | Value |
|---|---|
| Modal server ki jagah | **US West (Azure westus3)** |
| Tumhare machine se RTT | **~350 ms** |
| Ek connection pe speed (HTTP ya WebSocket dono) | **~1.3 – 2.4 Mbps** |
| 4 parallel connections | ~3.6 – 6.6 Mbps total |
| Tumhare laptop ki normal internet speed (Cloudflare se) | ~50 Mbps |
| Ek JPEG frame ka size | 77.5 KB |
| 25 fps ke liye chahiye | **~15.9 Mbps** |

**Layman example:**
- Factory ne 250 frames 15 second mein tayar kar diye, par truck ek baar mein bahut kam maal le ja pa raha tha. Isliye customer ko 73 second lage.
- Ye truck ki problem hai, factory ki nahi. Iska saboot: **bina GPU ke, sirf random bytes bhejne pe bhi wahi ~2 Mbps aaya.**
- India se US West ka rasta lamba hai (350 ms). Aise lambe rasta pe thodi si bhi packet loss ek connection ko ~1-2 Mbps pe atka deti hai. (Ye mera andaza hai, abhi verify nahi kiya.)
- Hum 15.9 Mbps ka maal 2 Mbps wali pipe se bhejne ki koshish kar rahe the.

### 5.6 Bottleneck ka nichod (H.264 se pehle ka)

```
Server ka speed:   ~16.5 fps   (thoda sudhar chahiye)
Network ka speed:   ~3.4 fps   ◄── ASLI RUKAVAT
Real-time target:    25 fps
```

### 5.7 Meri galat guesses (transparency ke liye)
- Pehle laga blending serial hone se slow hai → sirf 25 ms/frame nikla, chhota issue.
- Phir laga har frame ko alag message bhejne ka overhead hai → nahi, `send()` sirf 0.05 ms leta hai.
- Measure karne ke baad hi asli wajah (network bytes) pakdi gayi.

### 5.8 H.264 + parallel pipeline ke baad (10 sec audio, 250 frames, warm)

| Metric | Pehle (JPEG, serial) | Ab (H.264, parallel) |
|---|---|---|
| Video bitrate @25fps | 15.9 Mbps | **0.81 Mbps** (20x kam) |
| Frame size | 77.5 KB | **~4 KB** |
| Client ko poora clip milne mein | 72.8 s | **11.6 s** |
| Frames ka arrival rate (client pe) | 3.4 fps | **28.4 fps** (real-time = 25) |
| Server ka total time | 15.2 s (~16.5 fps) | **9.3 s (~28 fps)** |
| Pehla frame client pe | 2.3 s | 2.85 s (server 0.44 s + ~2.4 s network/audio upload) |
| Smooth playback ke liye startup buffer | — | **2.95 s** (play ke baad koi rukavat nahi) |

Cold run (pehli request): pehla frame **29.6 s** baad, isme 22 s audio-features ka one-time warm-up hai. Ye abhi fix nahi hua (Priority 3).

**Kya seekha:**
1. **Network ki dikkat H.264 ne hal kar di.** 0.81 Mbps us ~2 Mbps wali line pe bhi aaram se fit ho jata hai.
2. **Ab bottleneck server hai (~28 fps), network nahi.** Client ka arrival rate aur server ka speed lagbhag barabar hain.
3. **Parallel threads ne utna nahi diya jitna socha tha.** Total speed 16.5 → ~28 fps hui, par har stage individually slow ho gaya: GPU stage 11 → 37 ms/frame, blend 25 → 84 ms/frame. Wajah (mera andaza): Python ke threads ek hi "GIL" lock share karte hain. **Example:** 4 rasoiye hain, par gas ka chulha ek hi, isliye har rasoiye ki speed 3x kam ho gayi aur total sirf thoda badha.
4. 28 fps, 25 se sirf thoda upar hai. Koi margin nahi, isliye jitter aaya to playback atak sakta hai.

### 5.9 Fast blend ke baad (warm container)

| Metric | H.264 + PIL blend (5.8) | H.264 + fast blend (ab) |
|---|---|---|
| Blend per frame (server pe) | 84 ms | **13 ms** |
| GPU stage per frame (pipeline mein) | 37 ms | **22 ms** (akele 11 ms) |
| Server speed (10 s clip) | ~27 fps (9.3 s) | **~46 fps (5.5 s)** |
| Client pe frames ka arrival rate | 28.4 fps | **49.2 fps** |
| Client ko poora 10 s clip milne mein | 11.6 s | **7.9 s** |
| Frames late (play pehle frame pe dabaya to) | 2/250 | **0/250** |
| **60 s ka clip** (1500 frames) poora milne mein | 533 s (JPEG tha) | **39.5 s**, arrival 48 fps, 1/1500 frame late |

**Kya seekha:**
1. **Ab real-time hai, ~2x margin ke saath.** Generation playback se tez hai: 60 s ka video 32 s mein ban gaya. Isliye pehla frame aate hi play shuru kar sakte hain aur video kabhi atkega nahi.
2. **Ab bhi GPU thread hi pipeline ki speed tay karta hai** (22 ms/frame ≈ 46 fps), kyunki wo abhi bhi akele se 2x slow hai (CPU threads se). Aage ki gunjaish ~60-80 fps tak lagti hai (encoder ~11 ms/frame ka ceiling ~87 fps, mera andaza).
3. **Pehla frame ka intezaar ab network + audio upload hai, server nahi.** Server 0.35-0.37 s mein pehla packet bhej deta hai, par client ko 10 s clip pe 2.87 s aur 60 s clip pe 8.4 s lagta hai. 10 s (320 KB) aur 60 s (1.92 MB) ke numbers ko fit karne pe: ~1.4 s fixed + audio upload ~2.3 Mbps pe. Yani audio ka size seedha pehle frame ko late karta hai.
4. **Kabhi-kabhi ek chhota atka** (~0.46 s, 60 s clip mein ek baar). Isliye client pe ~0.5-1 s ka jitter buffer rakhna sahi rahega.
5. **Cold run abhi bhi ~43 s** (audio-features 31 s + pehla GPU batch 9 s, ye ek baar ka warm-up hai, run-to-run 22-31 s ghoomta hai). Ye agla bada kaam hai.

### 5.10 Asli server pe final run: 25 fps + audio (60 sec ka clip)

Poora chain: player (aapki machine) → internet → Modal A100 → wapas. Server ko pehle ek silent warm-up request se warm kiya gaya tha.

| Metric | Result |
|---|---|
| Frames draw hue | **1500 / 1500** |
| Playback speed | **25.00 fps** poore 60.0 s tak |
| Skipped / late frames | **0 / 0** |
| Draw timing error (frame apne due time se kitna late draw hua) | p50 2.2 ms, p95 4.3 ms, max 7.0 ms (ek frame = 40 ms) |
| Last frame pe hold (agla frame nahi aaya) | 0.00 s |
| Video bandwidth | 0.82 Mbps |
| Server ne 60 s ka video banaya | 34 s mein (~44 fps) |
| Client pe jama buffer (peak) | 694 packets = 27.8 s ka video (sirf ~3 MB) |
| Audio bhejne se pehli awaaz + frame tak | **8.8 s** (isme server ka hissa sirf 0.58 s) |

Warm-up (cold) request, 10 s audio: pehla packet **41 s** baad aaya, lekin playback phir bhi 25.01 fps, 0 skipped.

**Kya seekha:**
1. Playback ab **smooth aur real-time** hai. Server tez hai, isliye client peeche se buffer banata hai, aur network mein chhota atka aaye to bhi video nahi rukega.
2. Jo delay bacha hai wo **server ki nahi hai:** (a) cold start ~40 s, (b) audio upload (1.9 MB WAV slow link pe ≈ 8 s).
3. Awaaz aur hoth ka lip-sync **aankh-kaan se abhi check hona baaki hai.** Numbers sirf frame ki timing verify karte hain. Offset lage to `--av-offset-ms` se adjust karo.

---

## 6. Future plan (improvements)

### Priority 1 — Network fix (sabse bada asar)

| Kaam | Kyun | Expected asar |
|---|---|---|
| **GPU ko India ke paas chalana** (`region="ap-south"`, Mumbai) | RTT 350 ms se kam hokar tens of ms ho sakta hai | Throughput bahut badh sakta hai |
| ✅ **JPEG ki jagah H.264 video chunks** (ho gaya) | JPEG 15.9 Mbps tha | Measured: **0.81 Mbps**, 2 Mbps wali line pe bhi fit |
| Sirf **chehre ka chhota crop** bhejna, background client pe rakhna | Avatar ka background fixed hai, har baar bhejne ki zaroorat nahi | Data ~5-8x kam + server ka blend ka kaam bhi bachega |
| Server ko receiver ki speed ke hisaab se **pace** karna | Warna data pipe mein queue ho jata hai (keepalive timeout bhi isi se aaya) | Latency stable rahegi |

> Docs ke hisaab se `ap-south` (Mumbai) valid region hai. Price multiplier: broad region (`"ap"`) 1.15x, narrow (`"ap-south"`) 1.75x. **A100 Mumbai mein milega ya nahi, ye docs mein nahi likha, test karna padega.**
> Ye sab candidates ki location pe depend karta hai (agar zyadatar India mein hain to Mumbai theek hai).

**Example (H.264):** JPEG = har photo alag alag bhaari parcel mein bhejna. H.264 = video ko vacuum-pack karke bhejna, kyunki har frame pichle frame se milta-julta hai.

### Priority 2 — Server ko 25+ fps pe lana

| Kaam | Expected asar |
|---|---|
| ✅ GPU, blend, encode **parallel threads** mein + ✅ blend ko sasta banana (ho gaya) | Measured: 16.5 → 27 → **~48 fps**. Aur upar jaana ho to: GPU thread ko CPU threads se aur alag karna, ya blend ko GPU pe (avatar ke 384 frames ~1 GB, A100 ke liye kuch nahi). Abhi zaroorat nahi, margin theek hai |
| ✅ Container ko `cpu=8` diya (`server.py`) | Blend/encode CPU ka kaam hai. Alag se measure nahi kiya ki isse kitna fark pada |

**Example:** Pehle ek hi aadmi photo khinchta, edit karta aur print karta tha (59 ms). Ab teeno kaam assembly line mein alag-alag logon ke paas hain (GPU → blend → encoder), isliye output tez nikalta hai.

### Priority 3 — Cold start hatana

- Startup pe **dummy inference** (warm-up) chalana → pehle request ka ~28 s hat jayega.
- `min_containers=1` interview hours mein (paisa vs speed ka trade-off).
- `s3fd` face-detection model (85 MB) har cold start pe internet se download ho raha hai → Volume mein cache karna.

### Priority 4 — Client side (interview app)

- ✅ (test client mein ho gaya) Audio client pe bajana, frames **25 fps ke fixed pace** pe dikhana, chhota jitter buffer (0.5 s), audio master clock. Frame late ho to last frame hold, peeche reh jaye to frames skip. Ab interview app ke browser mein bhi yahi logic lagana hai.
- Video chunks ko browser mein **MediaSource Extensions** se chalana.
- Stall pe avatar ka "idle loop" dikhana taaki video freeze na lage.
- Audio upload chhota karna: WAV bhari hai (10 s = 320 KB, 60 s = 1.92 MB). Slow link pe 60 s ka clip pehle frame ko ~8 s late karta hai. Opus/MP3 mein bhejne se ~10x chhota hoga. **Ya better:** TTS (text-to-speech) ko GPU ke saath cloud mein hi rakho aur app sirf text bheje, to audio slow link se guzre hi nahi.

### Priority 5 — Production tayari

- `modal deploy` se permanent service (abhi sirf `modal serve`/`modal run` se test hua hai).
- Per-session **auth token** (warna koi bhi URL pe GPU ka paisa jala sakta hai).
- Session-to-container mapping, concurrency limit, monitoring (audio-aane se pehle-frame-tak ka latency).
- Audio ko **chhote chunks** mein process karna (jab TTS se audio streaming aa raha ho to poore audio ka wait na karna pade).
- Agar candidates alag-alag countries mein hain: multi-region ya WebRTC (UDP-based, lambe rasta pe TCP se behtar kaam karta hai — ye future option hai, abhi verify nahi kiya).

### Target numbers (ye hasil karne hain)

| Metric | Abhi | Target |
|---|---|---|
| Pehla frame (warm) | ~2.9 s (10 s audio), ~8.4 s (60 s audio), server sirf 0.35 s | **< 1.5 s** (audio compress / TTS cloud mein) |
| Pehla frame (cold) | ~30-43 s (measure hone pe ghoomta hai) | **< 3 s** (warm-up + min_containers se) |
| Sustained speed | **~48 fps** ✅ (60 s clip pe bhi) | **>= 35 fps** (achieved) |
| Bandwidth per user | ~0.81 Mbps ✅ | **1 – 2 Mbps** |
| Server per-frame time (effective) | **~21 ms** ✅ | **< 28 ms** (achieved) |
| Playback smoothness (asli server, awaaz ke saath, 60 s) | **1500/1500 frames, 0 skipped, 0 late** ✅ | 0 skipped (achieved) |

---

## 7. Commands (project chalane ke liye sab kuch)

> Project folder ke naam mein space hai (`avatar gpu test`), isliye path hamesha quotes mein likho. Har naye terminal mein pehle `source venv/bin/activate`.

### 7.1 Local setup (ek baar)

```bash
cd "/home/dhananjay/Public/avatar gpu test"
source venv/bin/activate

pip install modal websockets av opencv-python numpy sounddevice
sudo apt install libportaudio2      # sounddevice (audio bajane) ke liye, Linux pe

modal token new                     # Modal account se login (ek baar)
```

### 7.2 Cloud setup (ek baar, ya jab avatar/models badlein)

**Zaroori order:** avatar pehle upload karo, phir preprocess chalao.

```bash
# avatar video Volume mein upload
modal volume put musetalk-data sample.mp4 avatars/avatar.mp4

# 6 AI models download + avatar preprocess (dono ek command mein)
modal run preprocess.py

# check karo ki sab kuch Volume mein hai
modal volume ls musetalk-data models
modal volume ls musetalk-data avatars/results/v15/avatars/avatar_1
```

- `download_weights` marker ki wajah se dobara download nahi karta.
- `preprocess_avatar` har baar avatar ko naye sire se banata hai (~1.5 min GPU), isliye sirf avatar badalne pe chalao.
- Alag alag chalane ke liye: `modal run preprocess.py::download_weights` ya `modal run preprocess.py::preprocess_avatar`.
- Doosra avatar banana ho to `server.py` ka `AVATAR_ID` bhi badalna padega.

### 7.3 Roz ka flow

**Terminal 1: server**
```bash
modal serve server.py
# "Serving..." dikhe to ready. Band karne ke liye Ctrl-C.
# wss://dhananjay-sharma--musetalk-lipsync-musetalkinference-web-dev.modal.run/ws-stream   (H.264, streaming)
# wss://dhananjay-sharma--musetalk-lipsync-musetalkinference-web-dev.modal.run/ws          (poora mp4 ek saath)
```

**Terminal 2: player (25 fps + audio)**
```bash
python tools/ws_h264_view.py test_audio.wav
```
Pehli request pe cold start ~40 s lagta hai. Demo se pehle ek silent warm-up bhej do:
```bash
python tools/ws_h264_view.py test_audio.wav --no-display --mute
```

| Player option | Kaam |
|---|---|
| `--prebuffer 0.5` | Play shuru karne se pehle kitna video jama kare (jitter cushion) |
| `--av-offset-ms 0` | Lip-sync fine-tune (+ = video late, − = video pehle) |
| `--overlay` | Screen pe time aur buffer dikhao |
| `--mute` / `--no-audio` / `--no-display` | Testing ke liye |
| `--url wss://...` | Alag server URL |
| `--runs N` | Ek hi connection pe N baar chalao |
| `q` ya `Esc` | Window se band karna |

### 7.4 Baaki useful commands

```bash
# poora mp4 ek saath banana (MuseTalk ka sample audio, har baar cold start), output: test_output.mp4
modal run server.py

# agar test_audio.wav nahi hai to MuseTalk ka sample audio local pe le aao
modal run inspect_musetalk.py::fetch_audio

# 10 sec ka chhota audio banana (quick test ke liye)
ffmpeg -i test_audio.wav -t 10 audio10.wav

# pehla GPU sanity test
modal run test.py
```

### 7.5 Server band karna aur check karna (paisa bachane ke liye)

```bash
# serve foreground mein chal raha ho: Ctrl-C
# agar background mein chal raha ho:
pkill -INT -f "[v]env/bin/modal serve server.py"

modal app list                      # State "stopped" aur Tasks 0 hona chahiye
modal app stop <app-id>             # zabardasti band karna (deployed app ke liye)
```

### 7.6 Permanent deploy (abhi nahi kiya)

```bash
modal deploy server.py
```
Isse URL mein `-dev` nahi hoga aur service hamesha available rahegi. **Isse pehle auth token lagana zaroori hai**, warna URL jaanne wala koi bhi aapke GPU ka paisa jala sakta hai. Deploy ke baad player mein naya URL `--url` se dena hoga.

### 7.7 Yaad rakhne wali cheezein

- `modal run` har baar naya temporary app banata hai aur khatam hote hi band ho jata hai. "Warm container" ka fayda sirf `modal serve` / `modal deploy` mein milta hai.
- `modal serve` chalte waqt GPU container warm rehta hai aur **paisa lagta hai.** Kaam khatam hote hi band karo.
- Version pins (torch 2.0.1+cu118, diffusers 0.27.2, huggingface_hub 0.23.4, numpy 1.23.5, av 17.1.0) **badalna mat**, ye sab ek dusre se jude hain.
- Aam galtiyan: `venv` activate karna bhool jana (to `modal` ya `websockets` nahi milega), avatar upload kiye bina `modal run preprocess.py` chalana (`FileNotFoundError`), server chalu chhod dena.

---

## 8. Ek line mein poori kahani

Pipeline **end-to-end kaam karta hai:** audio jata hai, lip-synced avatar video H.264 mein aata hai, aur player use **25 fps pe awaaz ke saath** chalata hai. Asli server pe 60 s ka clip 1500/1500 frames, 0 skipped, 0 late ke saath chala. Server warm hone pe ~45-48 fps banata hai (real-time ka ~2x), bandwidth 15.9 → 0.82 Mbps ho gayi, aur blend 8x tez hua (bit-exact). **Baaki kaam:** (1) **cold start** hatana (pehli request pe ~40 s), (2) **pehle frame ki delay** kam karna (audio upload + 350 ms RTT ki wajah se, server ki wajah se nahi), (3) **server ko India ke paas rakhna** (`ap-south`, A100 availability check karni hai), (4) **deploy + auth** aur interview app (browser) ke saath integration.
