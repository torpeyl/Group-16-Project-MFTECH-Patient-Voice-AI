const recordButton = document.getElementById('recordButton');
const audioPlayback = document.getElementById('audioPlayback');
const timerElement = document.getElementById('timer');
const fileInput = document.getElementById('fileInput');
const fileNameDisplay = document.getElementById('fileName');

let mediaRecorder;
let audioChunks = [];
let timer;
let startTime;
let isRecording = false;

recordButton.addEventListener('click', async () => {
    if (isRecording) {
        mediaRecorder.stop();
        recordButton.style.backgroundColor = 'red';
        recordButton.classList.remove('recording');
    } else {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            audioChunks = [];
            clearInterval(timer);
            timerElement.textContent = '00:00.000';
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
    }
});

fileInput.addEventListener('change', event => {
    const file = event.target.files[0];
    if (file) {
        const audioUrl = URL.createObjectURL(file);
        audioPlayback.src = audioUrl;
        fileNameDisplay.textContent = `File: ${file.name}`;
    }
});

function updateTimer() {
    const elapsedTime = Date.now() - startTime;
    const totalSeconds = Math.floor(elapsedTime / 1000);
    const milliseconds = String(elapsedTime % 1000).padStart(3, '0');
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
    const seconds = String(totalSeconds % 60).padStart(2, '0');
    timerElement.textContent = `${minutes}:${seconds}.${milliseconds}`;
}
