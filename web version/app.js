// ============================================================
// SKETCHAIR WEB APP
// Browser Camera + Python WebSocket Engine
// ============================================================

"use strict";


// ============================================================
// BACKEND CONFIG
// ============================================================
//
// Leave BACKEND_HOST empty ("") when the frontend and backend
// are served from the same origin (e.g. running web_server.py
// locally, or hosting everything together on Render/Railway).
//
// Set it to your backend's host when the frontend is deployed
// separately (e.g. on Netlify) from the Python backend (e.g.
// on Render). Do NOT include "https://" or a trailing slash —
// just the host, like: "sketchair-backend.onrender.com"
// ============================================================

const BACKEND_HOST = "sketchair-936f.onrender.com";


// ============================================================
// DOM ELEMENTS
// ============================================================

const startScreen = document.getElementById("startScreen");
const app = document.getElementById("app");

const startButton = document.getElementById("startButton");
const startError = document.getElementById("startError");

const video = document.getElementById("video");

const drawingCanvas =
    document.getElementById("drawingCanvas");

const uiCanvas =
    document.getElementById("uiCanvas");

const modeBadge =
    document.getElementById("modeBadge");

const modeText =
    document.getElementById("modeText");

const colorPreview =
    document.getElementById("colorPreview");

const colorText =
    document.getElementById("colorText");

const sizeText =
    document.getElementById("sizeText");

const colorGrid =
    document.getElementById("colorGrid");

const recentColors =
    document.getElementById("recentColors");

const hoverPreview =
    document.getElementById("hoverPreview");

const hoverColorDot =
    document.getElementById("hoverColorDot");

const hoverColorName =
    document.getElementById("hoverColorName");

const undoButton =
    document.getElementById("undoButton");

const redoButton =
    document.getElementById("redoButton");

const decreaseSize =
    document.getElementById("decreaseSize");

const increaseSize =
    document.getElementById("increaseSize");

const sizeSlider =
    document.getElementById("sizeSlider");

const sizeValue =
    document.getElementById("sizeValue");

const clearButton =
    document.getElementById("clearButton");

const saveButton =
    document.getElementById("saveButton");

const toast =
    document.getElementById("toast");

const loadingOverlay =
    document.getElementById("loadingOverlay");


// ============================================================
// CANVAS CONTEXTS
// ============================================================

const drawingCtx =
    drawingCanvas
        ? drawingCanvas.getContext("2d")
        : null;

const uiCtx =
    uiCanvas
        ? uiCanvas.getContext("2d")
        : null;


// ============================================================
// APPLICATION STATE
// ============================================================

let cameraStream = null;

let socket = null;

let applicationRunning = false;

let frameTimer = null;

let frameSending = false;
let awaitingFrameReply = false;
let awaitingReplyTimer = null;
let requestNextFrame = null;

let toastTimer = null;

let currentColor = "RED";

let currentSize = 6;

let recentColorList = [];


// ============================================================
// COLORS
// ============================================================

const COLORS = {

    BLUE: "#245fbd",

    GREEN: "#247d3d",

    YELLOW: "#c9a900",

    ORANGE: "#c86c16",

    BROWN: "#70482d",

    BLACK: "#000000",

    WHITE: "#ffffff",

    RED: "#b52a2a",

    PURPLE: "#743a92",

    PINK: "#c44e83"

};


// ============================================================
// INITIALIZATION
// ============================================================

function initialize() {

    console.log(
        "SketchAir web application starting..."
    );

    console.log(
        "Browser MediaPipe: DISABLED"
    );

    console.log(
        "Python processing: WebSocket"
    );


    currentColor = "RED";

    currentSize = 6;


    updateColorUI();

    updateSizeUI();

    createPalette();

    setupButtons();

    setupKeyboard();

    resizeCanvases();

    window.addEventListener(
        "resize",
        resizeCanvases
    );


    setMode(
        "NO HAND",
        "idle"
    );


    if (loadingOverlay) {

        loadingOverlay.classList.add(
            "hidden"
        );

    }


    console.log(
        "SketchAir initialized successfully."
    );

}


// ============================================================
// RESIZE CANVASES
// ============================================================

function resizeCanvases() {

    if (!video) {
        return;
    }


    const width =
        video.videoWidth ||
        video.clientWidth ||
        window.innerWidth;

    const height =
        video.videoHeight ||
        video.clientHeight ||
        window.innerHeight;


    if (!width || !height) {
        return;
    }


    if (drawingCanvas) {

        if (
            drawingCanvas.width !== width ||
            drawingCanvas.height !== height
        ) {

            drawingCanvas.width = width;
            drawingCanvas.height = height;

        }

    }


    if (uiCanvas) {

        if (
            uiCanvas.width !== width ||
            uiCanvas.height !== height
        ) {

            uiCanvas.width = width;
            uiCanvas.height = height;

        }

    }

}


// ============================================================
// CAMERA
// ============================================================

async function startCamera() {

    if (!navigator.mediaDevices) {

        throw new Error(
            "Camera API is not available in this browser."
        );

    }


    if (
        !navigator.mediaDevices.getUserMedia
    ) {

        throw new Error(
            "Your browser does not support webcam access."
        );

    }


    cameraStream =
        await navigator.mediaDevices.getUserMedia({

            video: {

                width: {
                    ideal: 1280
                },

                height: {
                    ideal: 720
                },

                facingMode: "user"

            },

            audio: false

        });


    if (!video) {

        throw new Error(
            "Video element was not found."
        );

    }


    video.srcObject =
        cameraStream;


    await video.play();


    resizeCanvases();


    console.log(
        "Browser camera started."
    );

}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

    }


    cameraStream = null;


    if (video) {

        video.srcObject = null;

    }

}


// ============================================================
// START APPLICATION
// ============================================================

async function startApplication() {

    if (applicationRunning) {
        return;
    }


    console.log(
        "Start button clicked."
    );


    if (startButton) {

        startButton.disabled = true;

        startButton.textContent =
            "Starting…";

    }


    clearError();


    try {

        // ----------------------------------------
        // 1. Start browser camera
        // ----------------------------------------

        await startCamera();


        // ----------------------------------------
        // 2. Show drawing application
        // ----------------------------------------

        if (startScreen) {

            startScreen.classList.add(
                "hidden"
            );

        }


        if (app) {

            app.classList.remove(
                "hidden"
            );

        }


        applicationRunning = true;


        setMode(
            "CAMERA READY",
            "active"
        );


        // ----------------------------------------
        // 3. Connect to Python
        // ----------------------------------------

        showToast(
            "Camera started"
        );


        connectToPython();


    } catch (error) {

        console.error(
            "START ERROR:",
            error
        );


        stopCamera();


        applicationRunning = false;


        showError(
            getCameraErrorMessage(error)
        );


        if (startButton) {

            startButton.disabled = false;

            startButton.textContent =
                "Start Air Drawing";

        }

    }

}


// ============================================================
// CAMERA ERROR MESSAGE
// ============================================================

function getCameraErrorMessage(error) {

    if (!error) {

        return "Unable to start camera.";

    }


    if (
        error.name ===
        "NotAllowedError"
    ) {

        return (
            "Camera permission was blocked. " +
            "Allow camera access in Chrome and try again."
        );

    }


    if (
        error.name ===
        "NotFoundError"
    ) {

        return (
            "No camera was found on this computer."
        );

    }


    if (
        error.name ===
        "NotReadableError"
    ) {

        return (
            "The camera is already being used by another application."
        );

    }


    return (
        "Camera error: " +
        error.message
    );

}


// ============================================================
// ERROR DISPLAY
// ============================================================

function showError(message) {

    if (!startError) {
        return;
    }


    startError.textContent =
        message;


    startError.classList.add(
        "visible"
    );

}


function clearError() {

    if (!startError) {
        return;
    }


    startError.textContent = "";


    startError.classList.remove(
        "visible"
    );

}


// ============================================================
// CONNECT TO PYTHON SERVER
// ============================================================

function connectToPython() {

    if (socket) {

        try {

            socket.close();

        } catch (_) {}

    }


    const protocol =
        location.protocol === "https:"
            ? "wss:"
            : "ws:";


    const host =
        BACKEND_HOST || location.host;


    const wsUrl =
        `${protocol}//${host}/ws`;


    console.log(
        "Connecting to Python:",
        wsUrl
    );


    setMode(
        "CONNECTING",
        "active"
    );


    socket =
        new WebSocket(wsUrl);


    socket.binaryType =
        "blob";


    // ----------------------------------------
    // OPEN
    // ----------------------------------------

    socket.onopen = () => {

        console.log(
            "Python WebSocket connected."
        );


        setMode(
            "READY",
            "active"
        );


        hideLoading();


        showToast(
            "Python engine connected"
        );


        startFrameLoop();

    };


    // ----------------------------------------
    // MESSAGE
    // ----------------------------------------

    socket.onmessage =
        handlePythonMessage;


    // ----------------------------------------
    // ERROR
    // ----------------------------------------

    socket.onerror =
        error => {

            console.error(
                "Python WebSocket error:",
                error
            );


            setMode(
                "PYTHON ERROR",
                "error"
            );


            showToast(
                "Python connection error"
            );

        };


    // ----------------------------------------
    // CLOSE
    // ----------------------------------------

    socket.onclose =
        () => {

            console.log(
                "Python WebSocket disconnected."
            );


            stopFrameLoop();


            if (applicationRunning) {

                setMode(
                    "PYTHON OFFLINE",
                    "error"
                );


                showToast(
                    "Python server disconnected"
                );

            }

        };

}


// ============================================================
// PYTHON MESSAGE HANDLER
// ============================================================

function handlePythonMessage(event) {


    // ========================================================
    // JSON MESSAGE
    // ========================================================

    if (
        typeof event.data ===
        "string"
    ) {

        try {

            const data =
                JSON.parse(
                    event.data
                );


            handlePythonState(
                data
            );


        } catch (error) {

            console.warn(
                "Invalid Python JSON:",
                error
            );

        }


        return;

    }


    // ========================================================
    // IMAGE MESSAGE
    // ========================================================

    if (
        event.data instanceof Blob
    ) {

        awaitingFrameReply = false;

        if (awaitingReplyTimer) {
            clearTimeout(awaitingReplyTimer);
            awaitingReplyTimer = null;
        }

        displayProcessedFrame(
            event.data
        );

        if (requestNextFrame) {
            requestNextFrame();
        }

    }

}


// ============================================================
// PYTHON STATE
// ============================================================

function handlePythonState(data) {

    if (!data) {
        return;
    }


    // ----------------------------------------
    // Gesture
    // ----------------------------------------

    if (
        data.mode !== undefined
    ) {

        setMode(
            String(data.mode),
            "active"
        );

    }


    if (
        data.gesture !== undefined
    ) {

        setMode(
            String(data.gesture),
            "active"
        );

    }


    // ----------------------------------------
    // Color
    // ----------------------------------------

    if (
        data.color
    ) {

        setColorFromPython(
            data.color
        );

    }


    // ----------------------------------------
    // Brush size
    // ----------------------------------------

    if (
        data.brushSize !== undefined
    ) {

        const size =
            Number(
                data.brushSize
            );


        if (
            Number.isFinite(size)
        ) {

            currentSize =
                Math.max(
                    2,
                    Math.min(
                        30,
                        size
                    )
                );


            updateSizeUI();

        }

    }


    // ----------------------------------------
    // Message
    // ----------------------------------------

    if (
        data.message
    ) {

        showToast(
            String(data.message)
        );

    }


    // ----------------------------------------
    // Selected color message
    // ----------------------------------------

    if (
        data.selectedColor
    ) {

        setColorFromPython(
            data.selectedColor
        );


        showToast(
            `${data.selectedColor} SELECTED`
        );

    }

}


// ============================================================
// DISPLAY PROCESSED PYTHON FRAME
// ============================================================

function displayProcessedFrame(blob) {

    if (!uiCanvas || !uiCtx) {
        return;
    }


    const image =
        new Image();


    image.onload = () => {

        resizeCanvases();


        uiCtx.clearRect(
            0,
            0,
            uiCanvas.width,
            uiCanvas.height
        );


        uiCtx.drawImage(
            image,
            0,
            0,
            uiCanvas.width,
            uiCanvas.height
        );


        URL.revokeObjectURL(
            image.src
        );

    };


    image.onerror = () => {

        URL.revokeObjectURL(
            image.src
        );

    };


    image.src =
        URL.createObjectURL(
            blob
        );

}


// ============================================================
// CAMERA FRAME LOOP
// ============================================================

function startFrameLoop() {

    stopFrameLoop();


    if (!video) {
        return;
    }


    if (!drawingCanvas) {
        return;
    }


    const width = 480;

    const height = 270;


    drawingCanvas.width =
        width;

    drawingCanvas.height =
        height;


    if (uiCanvas) {

        uiCanvas.width =
            width;

        uiCanvas.height =
            height;

    }


    const ctx =
        drawingCanvas.getContext(
            "2d"
        );


    if (!ctx) {
        return;
    }


    function sendFrame() {

        if (!applicationRunning) {
            return;
        }


        if (
            !socket ||
            socket.readyState !==
                WebSocket.OPEN
        ) {

            frameTimer =
                setTimeout(
                    sendFrame,
                    100
                );

            return;

        }


        if (
            video.readyState < 2
        ) {

            frameTimer =
                setTimeout(
                    sendFrame,
                    100
                );

            return;

        }


        if (
            frameSending ||
            awaitingFrameReply
        ) {

            frameTimer =
                setTimeout(
                    sendFrame,
                    60
                );

            return;

        }


        // Prevent excessive WebSocket buffering

        if (
            socket.bufferedAmount >
            400000
        ) {

            frameTimer =
                setTimeout(
                    sendFrame,
                    100
                );

            return;

        }


        frameSending = true;


        // Draw current camera frame

        ctx.drawImage(
            video,
            0,
            0,
            width,
            height
        );


        // Convert to JPEG

        drawingCanvas.toBlob(
            blob => {

                if (
                    blob &&
                    socket &&
                    socket.readyState ===
                        WebSocket.OPEN
                ) {

                    try {

                        socket.send(
                            blob
                        );

                    } catch (error) {

                        console.error(
                            "Frame send error:",
                            error
                        );

                    }

                }


                frameSending =
                    false;


                // Wait for the server's reply to THIS frame
                // before sending the next one, instead of
                // blindly firing every 60ms. On a slow/loaded
                // backend, sending on a fixed timer regardless
                // of round-trip time causes frames to queue up
                // faster than the server can process them, so
                // what you see on screen keeps falling further
                // behind real time. A short safety timeout
                // still applies in case a reply is ever lost.

                awaitingFrameReply = true;

                if (awaitingReplyTimer) {
                    clearTimeout(awaitingReplyTimer);
                }

                awaitingReplyTimer =
                    setTimeout(
                        () => {
                            awaitingFrameReply = false;
                            sendFrame();
                        },
                        1500
                    );

            },
            "image/jpeg",
            0.6
        );

    }


    requestNextFrame = sendFrame;


    sendFrame();

}


// ============================================================
// STOP FRAME LOOP
// ============================================================

function stopFrameLoop() {

    if (frameTimer) {

        clearTimeout(
            frameTimer
        );

    }


    if (awaitingReplyTimer) {

        clearTimeout(
            awaitingReplyTimer
        );

    }


    frameTimer = null;
    awaitingReplyTimer = null;
    frameSending = false;
    awaitingFrameReply = false;
    requestNextFrame = null;

}


// ============================================================
// CLOSE APPLICATION
// ============================================================

function stopApplication() {

    console.log(
        "Stopping SketchAir..."
    );


    applicationRunning =
        false;


    stopFrameLoop();


    if (socket) {

        try {

            socket.close();

        } catch (_) {}

    }


    socket = null;


    stopCamera();


    if (uiCtx && uiCanvas) {

        uiCtx.clearRect(
            0,
            0,
            uiCanvas.width,
            uiCanvas.height
        );

    }


    if (startScreen) {

        startScreen.classList.remove(
            "hidden"
        );

    }


    if (app) {

        app.classList.add(
            "hidden"
        );

    }


    if (startButton) {

        startButton.disabled =
            false;

        startButton.textContent =
            "Start Air Drawing";

    }


    setMode(
        "NO HAND",
        "idle"
    );

}


// ============================================================
// PALETTE
// ============================================================

function createPalette() {

    if (!colorGrid) {
        return;
    }


    colorGrid.innerHTML = "";


    Object.keys(COLORS)
        .forEach(
            colorName => {

                const button =
                    document.createElement(
                        "button"
                    );


                button.type =
                    "button";


                button.className =
                    "color-swatch";


                button.dataset.color =
                    colorName;


                button.title =
                    colorName;


                button.setAttribute(
                    "aria-label",
                    colorName
                );


                button.style.background =
                    COLORS[colorName];


                // White needs a visible border

                if (
                    colorName ===
                    "WHITE"
                ) {

                    button.style.border =
                        "3px solid #777";

                }


                button.addEventListener(
                    "click",
                    () => {

                        selectColor(
                            colorName
                        );

                    }
                );


                colorGrid.appendChild(
                    button
                );

            }
        );


    updatePaletteSelection();

}


// ============================================================
// COLOR SELECTION
// ============================================================

function selectColor(colorName) {

    if (
        !COLORS[colorName]
    ) {

        return;

    }


    currentColor =
        colorName;


    updateColorUI();


    addRecentColor(
        colorName
    );


    updatePaletteSelection();


    sendCommand({

        command:
            "select_color",

        color:
            colorName

    });


    showToast(
        `${colorName} SELECTED`
    );

}


// ============================================================
// COLOR FROM PYTHON
// ============================================================

function setColorFromPython(
    colorName
) {

    const normalized =
        String(
            colorName
        ).toUpperCase();


    if (
        !COLORS[normalized]
    ) {

        return;

    }


    currentColor =
        normalized;


    updateColorUI();

    addRecentColor(
        normalized
    );

    updatePaletteSelection();

}


// ============================================================
// COLOR UI
// ============================================================

function updateColorUI() {

    if (colorPreview) {

        colorPreview.style.background =
            COLORS[currentColor];

    }


    if (colorText) {

        colorText.textContent =
            currentColor;

    }


    updatePaletteSelection();

}


// ============================================================
// PALETTE SELECTION UI
// ============================================================

function updatePaletteSelection() {

    if (!colorGrid) {
        return;
    }


    colorGrid
        .querySelectorAll(
            ".color-swatch"
        )
        .forEach(
            button => {

                const selected =
                    button.dataset.color ===
                    currentColor;


                button.classList.toggle(
                    "selected",
                    selected
                );


                if (selected) {

                    button.setAttribute(
                        "aria-pressed",
                        "true"
                    );

                } else {

                    button.setAttribute(
                        "aria-pressed",
                        "false"
                    );

                }

            }
        );

}


// ============================================================
// RECENT COLORS
// ============================================================

function addRecentColor(
    colorName
) {

    recentColorList =
        recentColorList.filter(
            color =>
                color !== colorName
        );


    recentColorList.unshift(
        colorName
    );


    recentColorList =
        recentColorList.slice(
            0,
            4
        );


    renderRecentColors();

}


// ============================================================
// RENDER RECENT COLORS
// ============================================================

function renderRecentColors() {

    if (!recentColors) {
        return;
    }


    recentColors.innerHTML =
        "";


    recentColorList
        .forEach(
            colorName => {

                const button =
                    document.createElement(
                        "button"
                    );


                button.type =
                    "button";


                button.className =
                    "recent-color";


                button.title =
                    colorName;


                button.setAttribute(
                    "aria-label",
                    colorName
                );


                button.style.background =
                    COLORS[colorName];


                if (
                    colorName ===
                    "WHITE"
                ) {

                    button.style.border =
                        "2px solid #777";

                }


                button.addEventListener(
                    "click",
                    () => {

                        selectColor(
                            colorName
                        );

                    }
                );


                recentColors.appendChild(
                    button
                );

            }
        );

}


// ============================================================
// BRUSH SIZE
// ============================================================

function updateSizeUI() {

    currentSize =
        Math.round(
            currentSize
        );


    if (sizeSlider) {

        sizeSlider.value =
            currentSize;

    }


    if (sizeValue) {

        sizeValue.textContent =
            currentSize;

    }


    if (sizeText) {

        sizeText.textContent =
            `${currentSize} px`;

    }


    sendCommand({

        command:
            "brush_size",

        size:
            currentSize

    });

}


// ============================================================
// CHANGE BRUSH SIZE
// ============================================================

function changeBrushSize(
    amount
) {

    currentSize =
        Math.max(
            2,
            Math.min(
                30,
                currentSize + amount
            )
        );


    updateSizeUI();

}


// ============================================================
// MODE
// ============================================================

function setMode(
    text,
    state = "idle"
) {

    if (modeText) {

        modeText.textContent =
            text;

    }


    if (modeBadge) {

        modeBadge.classList.remove(
            "idle",
            "active",
            "error"
        );


        modeBadge.classList.add(
            state
        );

    }

}


// ============================================================
// TOAST
// ============================================================

function showToast(message) {

    if (!toast) {
        return;
    }


    toast.textContent =
        message;


    toast.classList.add(
        "visible"
    );


    if (toastTimer) {

        clearTimeout(
            toastTimer
        );

    }


    toastTimer =
        setTimeout(
            () => {

                toast.classList.remove(
                    "visible"
                );

            },
            1600
        );

}


// ============================================================
// LOADING
// ============================================================

function hideLoading() {

    if (!loadingOverlay) {
        return;
    }


    loadingOverlay.classList.add(
        "hidden"
    );

}


// ============================================================
// SEND COMMAND TO PYTHON
// ============================================================

function sendCommand(data) {

    if (
        !socket ||
        socket.readyState !==
            WebSocket.OPEN
    ) {

        console.warn(
            "Python is not connected.",
            data
        );

        return;

    }


    try {

        socket.send(
            JSON.stringify(
                data
            )
        );

    } catch (error) {

        console.error(
            "Command send error:",
            error
        );

    }

}


// ============================================================
// BUTTON SETUP
// ============================================================

function setupButtons() {


    // ----------------------------------------
    // START
    // ----------------------------------------

    if (startButton) {

        startButton.addEventListener(
            "click",
            startApplication
        );

    }


    // ----------------------------------------
    // UNDO
    // ----------------------------------------

    if (undoButton) {

        undoButton.addEventListener(
            "click",
            () => {

                sendCommand({
                    command: "undo"
                });

                showToast(
                    "UNDO"
                );

            }
        );

    }


    // ----------------------------------------
    // REDO
    // ----------------------------------------

    if (redoButton) {

        redoButton.addEventListener(
            "click",
            () => {

                sendCommand({
                    command: "redo"
                });

                showToast(
                    "REDO"
                );

            }
        );

    }


    // ----------------------------------------
    // DECREASE
    // ----------------------------------------

    if (decreaseSize) {

        decreaseSize.addEventListener(
            "click",
            () => {

                changeBrushSize(
                    -1
                );

            }
        );

    }


    // ----------------------------------------
    // INCREASE
    // ----------------------------------------

    if (increaseSize) {

        increaseSize.addEventListener(
            "click",
            () => {

                changeBrushSize(
                    1
                );

            }
        );

    }


    // ----------------------------------------
    // SLIDER
    // ----------------------------------------

    if (sizeSlider) {

        sizeSlider.addEventListener(
            "input",
            () => {

                currentSize =
                    Number(
                        sizeSlider.value
                    );


                updateSizeUI();

            }
        );

    }


    // ----------------------------------------
    // CLEAR
    // ----------------------------------------

    if (clearButton) {

        clearButton.addEventListener(
            "click",
            () => {

                sendCommand({
                    command: "clear"
                });


                if (
                    drawingCtx &&
                    drawingCanvas
                ) {

                    drawingCtx.clearRect(
                        0,
                        0,
                        drawingCanvas.width,
                        drawingCanvas.height
                    );

                }


                if (
                    uiCtx &&
                    uiCanvas
                ) {

                    uiCtx.clearRect(
                        0,
                        0,
                        uiCanvas.width,
                        uiCanvas.height
                    );

                }


                showToast(
                    "DRAWING CLEARED"
                );

            }
        );

    }


    // ----------------------------------------
    // SAVE
    // ----------------------------------------

    if (saveButton) {

        saveButton.addEventListener(
            "click",
            saveDrawing
        );

    }

}


// ============================================================
// KEYBOARD CONTROLS
// ============================================================

function setupKeyboard() {

    document.addEventListener(
        "keydown",
        event => {

            if (
                event.target &&
                (
                    event.target.tagName ===
                    "INPUT" ||
                    event.target.tagName ===
                    "TEXTAREA"
                )
            ) {

                return;

            }


            const key =
                event.key.toLowerCase();


            // Z = Undo

            if (key === "z") {

                sendCommand({
                    command: "undo"
                });

            }


            // X = Redo

            else if (key === "x") {

                sendCommand({
                    command: "redo"
                });

            }


            // C = Clear

            else if (key === "c") {

                sendCommand({
                    command: "clear"
                });

            }


            // + = Increase

            else if (
                key === "+" ||
                key === "="
            ) {

                changeBrushSize(
                    1
                );

            }


            // - = Decrease

            else if (
                key === "-"
            ) {

                changeBrushSize(
                    -1
                );

            }

        }
    );

}


// ============================================================
// SAVE DRAWING
// ============================================================

function saveDrawing() {

    if (!drawingCanvas) {

        showToast(
            "NOTHING TO SAVE"
        );

        return;

    }


    const link =
        document.createElement(
            "a"
        );


    link.download =
        `sketchair-${Date.now()}.png`;


    link.href =
        drawingCanvas.toDataURL(
            "image/png"
        );


    link.click();


    showToast(
        "DRAWING SAVED"
    );

}


// ============================================================
// CLEANUP
// ============================================================

window.addEventListener(
    "beforeunload",
    () => {

        applicationRunning =
            false;


        stopFrameLoop();

        stopCamera();


        if (socket) {

            try {

                socket.close();

            } catch (_) {}

        }

    }
);


// ============================================================
// START
// ============================================================

initialize();
