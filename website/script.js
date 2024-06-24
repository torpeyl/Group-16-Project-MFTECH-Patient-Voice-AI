const recordButton = document.getElementById('recordButton');
const audioPlayback = document.getElementById('audioPlayback');
const timerElement = document.getElementById('timer');
const fileInput = document.getElementById('fileInput');
const fileNameDisplay = document.getElementById('fileName');

const pain_slider = document.getElementById("pain-level-slider");
const pain_slider_label = document.getElementById("pain-level-label");
const submit_btn = document.getElementById('submit-btn');
const load_screen = document.getElementById('load-screen')

let mediaRecorder;
let audioChunks = [];
let timer;
let startTime;
let isRecording = false;

pain_slider.addEventListener('input', update_pain_label);

window.onload = (event) => {
    update_pain_label();
};

recordButton.addEventListener('click', async () => {
    if (isRecording) {      // Stop the recording

        mediaRecorder.stop();
        recordButton.style.backgroundColor = 'red';
        recordButton.classList.remove('recording');
        cancelAnimationFrame(animationFrameId);             // Stop the visualizer drawing loop

    } else {                // Start recording
        timerElement.textContent = '00:00.000';
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then(stream => {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                source = audioContext.createMediaStreamSource(stream);
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 2048;
                bufferLength = analyser.frequencyBinCount;
                dataArray = new Uint8Array(bufferLength);
                source.connect(analyser);

                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = e => {
                    audioChunks.push(e.data);
                };
                mediaRecorder.onstop = () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayback.src = audioUrl;
                    audio = new Audio(audioUrl);
                    audioChunks = [];                       // Clear the audioChunks array for future recordings
                    clearInterval(timer);
                    isRecording = false;
                };
                mediaRecorder.start();
                startTime = Date.now();
                timer = setInterval(updateTimer, 10);
                recordButton.style.backgroundColor = 'darkred';
                recordButton.classList.add('recording');
                isRecording = true;

                // Clear file name display when recording starts
                fileNameDisplay.textContent = '';

                draw();
            })
            .catch(err => {
                console.error('Error accessing the microphone', err);
            });
    }
});

fileInput.addEventListener('change', event => {
    const file = event.target.files[0];
    if (file) {
        const audioUrl = URL.createObjectURL(file);
        audioPlayback.src = audioUrl;
        fileNameDisplay.textContent = `File: ${file.name}`;
        timerElement.textContent = '00:00.000';
    }
});

submit_btn.addEventListener('click', event => {
    load_screen.style.display = "flex";
})

function updateTimer() {
    const elapsedTime = Date.now() - startTime;
    const totalSeconds = Math.floor(elapsedTime / 1000);
    const milliseconds = String(elapsedTime % 1000).padStart(3, '0');
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
    const seconds = String(totalSeconds % 60).padStart(2, '0');
    timerElement.textContent = `${minutes}:${seconds}.${milliseconds}`;
}

function draw() {
    animationFrameId = requestAnimationFrame(draw);

    if (!analyser) return;

    analyser.getByteTimeDomainData(dataArray);

    const canvas = document.getElementById('visualizer');
    const canvasCtx = canvas.getContext('2d');

    canvas.width = 400;
    canvas.height = 60;

    canvasCtx.fillStyle = 'rgb(240, 240, 240)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

    canvasCtx.beginPath();

    const sliceWidth = canvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;

        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

function update_pain_label() {
    pain_slider_label.innerHTML = 'Level of Pain: ' + pain_slider.value;
}
