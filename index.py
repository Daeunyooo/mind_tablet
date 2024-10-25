from flask import Flask, request, jsonify, make_response, render_template_string, session
import requests
import base64
import openai
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = os.environ.get('OPENAI_API_KEY')

# Define the fixed set of colors that can be used in the brush
BRUSH_COLORS = {
    '#f44336': 'red',
    '#ff5800': 'orange',
    '#faab09': 'yellow',
    '#008744': 'green',
    '#0057e7': 'blue',
    '#a200ff': 'purple',
    '#ff00c1': 'pink',
    '#ffffff': 'white',
    '#646765': 'grey',
    '#000000': 'black'
}

@app.route('/proxy')
def proxy_image():
    image_url = request.args.get('url')
    response = requests.get(image_url)
    proxy_response = make_response(response.content)
    proxy_response.headers['Content-Type'] = 'image/jpeg'
    proxy_response.headers['Access-Control-Allow-Origin'] = '*'
    return proxy_response

@app.route('/api/process-drawing', methods=['POST'])
def api_process_drawing():
    try:
        data = request.get_json()
        drawing_data = data['drawing']
        text_description = data['description']

        # Decode image from base64
        image_data = base64.b64decode(drawing_data.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('RGBA')

        # Extract colors used in the drawing
        raw_colors = {(r, g, b) for r, g, b, a in image.getdata() if a > 0}
        raw_colors_hex = {f"#{r:02x}{g:02x}{b:02x}" for r, g, b in raw_colors}
        used_colors_names = [BRUSH_COLORS[hex_color] for hex_color in raw_colors_hex if hex_color in BRUSH_COLORS]

        # Generate prompt using colors and description
        prompt = generate_prompt(text_description, used_colors_names)
        print(f"Generated prompt for DALL-E: {prompt}")

        # Generate image using the DALL-E API
        image_urls = call_dalle_api(prompt, n=2)
        if not image_urls:
            raise ValueError("Failed to generate images")

        # Generate reappraisal advice text
        reappraisal_text = generate_reappraisal_text(text_description)
        print(f"Generated reappraisal text: {reappraisal_text}")

        return jsonify({'image_urls': image_urls, 'reappraisal_text': reappraisal_text})
    except Exception as e:
        print(f"Error processing drawing: {str(e)}")
        return jsonify({'error': str(e)}), 500


def generate_prompt(description, colors=None):
    if colors:
        color_description = ', '.join(colors)
        prompt = (
            f"Create a purely visual artistic oil painting drawing using the colors {color_description}, "
            f"that reimagines '{description}' in a positive manner. For example, transforming a gloomy cloud "
            f"into a scene with a rainbow. The image must focus entirely on visual elements without any text, "
            f"letters, or numbers."
        )
    else:
        prompt = (
            f"Create a purely visual artistic oil painting drawing that reimagines '{description}' in a positive manner. "
            f"For example, transforming a gloomy cloud into a scene with a rainbow. The image must focus entirely "
            f"on visual elements without any text, letters, or numbers."
        )
    return prompt

def generate_reappraisal_text(description):
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Generate a short positive cognitive reappraisal advice for a child's description, less than three sentences: {description}",
            max_tokens=80
        )
        if 'choices' in response and len(response.choices) > 0:
            return response.choices[0].text.strip()
        else:
            return "Failed to generate meaningful output. Please refine the prompt."
    except Exception as e:
        print(f"Error generating reappraisal text: {str(e)}")
        return "Could not generate reappraisal text."


def call_dalle_api(prompt, n=2):
    api_key = app.secret_key
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "n": n, "size": "512x512"}

    try:
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        images = response.json().get('data', [])
        if not images:
            print("No images returned from DALL-E.")
        return [image['url'] for image in images]
    except requests.exceptions.RequestException as e:
        print(f"Error from OpenAI API: {e}")
        return []


predefined_sentences = {
    4: "Let's draw. Please use 'Visual Metaphor' on the right.",
    5: "Let's draw. Please use 'Visual Metaphor' on the right.",
    6: "Thank you for participating in the session. You can restart the session if you want to explore more."
}


def generate_art_therapy_question(api_key, question_number, session_history):
    openai.api_key = api_key
    question_prompts = [
        "Generate a question to ask user (children) about their current emotion. Do not use 'kiddo'.",
        "Based on the previous responses, generate a short question for identifying and describing the emotion, such as asking about the intensity of the emotion or where in the body it is felt the most. Users are kids, so please use easy and friendly expressions.",
        "Based on the previous responses, generate a short question that explores the context, such as asking what triggered this emotion or describing the situation or thought that led to these feelings. Users are kids, so please use easy and friendly expressions.",
        "Based on the previous responses, generate a short question that asks the user to describe and visualize their emotion as an 'abstract shape or symbol' to create their own metaphor for their mind. Users are kids, so please use easy and friendly expressions, and provide some metaphors or examples.",
        "Based on the previous responses, generate a short question that asks the user to describe and visualize their emotions as a 'texture' to create their own metaphor for their mind. Users are kids, so please use easy and friendly expressions, and provide some metaphors or examples.",
        "Based on the previous responses, provide personalized cognitive reappraisal advice to help think about the situation that user described in the previous response in a more positive way. Or, if user's previous response was already positive, please assist user to think about the good things they might learn from this experience. Please incorporating a playful and engaging approach consistent with CBT theory. Make sure the advice is directly relevant to the emotions and situations described by the child, using examples or activities that are fun and easy for kids to understand. Also, make this less than three sentences."
    ]
    
    user_responses = " ".join([resp for who, resp in session_history if who == 'You'])
    context = f"Based on the user's previous responses: {user_responses}"

    if 1 <= question_number <= 6:
        prompt_text = f"{context} {question_prompts[question_number - 1]}"
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt_text,
            max_tokens=150,
            n=1,
            temperature=0.7
        )
        question_text = response.choices[0].text.strip()

        if question_number in predefined_sentences:
            full_question_text = f"Question {question_number}: {predefined_sentences[question_number]} {question_text}"
        else:
            full_question_text = f"Question {question_number}: {question_text}"

        return full_question_text
    else:
        return "Do you want to restart the session?"




@app.route('/api/question', methods=['POST'])
def api_question():
    data = request.json
    user_response = data.get('response', '')
    session['history'] = session.get('history', [])
    session['responses'] = session.get('responses', [])
    session['question_number'] = session.get('question_number', 1)

    # Store the user's response
    session['history'].append(('You', user_response))
    session['responses'].append(user_response)

    if session['question_number'] <= 6:
        question_text = generate_art_therapy_question(
            app.secret_key, session['question_number'], session['history']
        )
        session['history'].append(('Therapist', question_text))
        session['question_number'] += 1
        progress = (session['question_number'] - 1) / 6 * 100
        return jsonify({
            'question': question_text,
            'progress': progress,
            'responses': session['responses'],
            'restart': False
        })
    else:
        # Send all responses back when it's the last question
        all_responses = "\n".join([f"Response {i+1}: {response}" for i, response in enumerate(session['responses'])])
        final_advice = generate_reappraisal_text(session['responses'][-1])
        session.clear()
        return jsonify({
            'question': 'Let\'s restart!',
            'progress': 100,
            'responses': all_responses + f"\nFinal Advice: {final_advice}",
            'restart': True
        })
        

@app.route('/', methods=['GET'])
def home():
    session['history'] = session.get('history', [])
    session['question_number'] = session.get('question_number', 1)
    initial_question = generate_art_therapy_question(
        app.secret_key, session['question_number'], session['history']
    )
    session['history'].append(('Therapist', initial_question))
    session['question_number'] += 1

    latest_question = session['history'][-1][1]
    progress_value = (session['question_number'] - 1) / 6 * 100
    return render_template_string("""
    <html>
        <head>
            <title>Mind Palette for kids!</title>
            <style>
                body {
                    font-family: 'Helvetica', sans-serif;
                    margin: 0;
                    padding: 0;
                }
                .container {
                    display: flex;
                    width: 100%;
                }
                .left, .right {
                    width: 50%;
                    padding: 20px;
                }
                .divider {
                    background-color: black;
                    width: 2px;
                    margin: 0 20px;
                    height: auto;
                }
                .active-tool {
                    background-color: black;
                    color: white;
                }
                .button-style {
                    color: white;
                    background-color: black;
                    padding: 5px 10px;
                    cursor: pointer;
                    border: none;
                    margin-left: 10px;
                    border-radius: 4px; 
                }
                .helper-text {
                    font-size: 18px; /* Set font size to 18px */
                    line-height: 1.6; /* Adjust line height for better readability */
                    color: black; /* Ensure the text is in black color */
                }
                #question {
                    font-size: 18px; /* Increase the font size for better readability */
                    line-height: 1.6; /* Adjust line height to add more space between lines */
                    margin-bottom: 20px; /* Additional margin below the text for spacing */
                    color: black; 
                }
                
                    progress {
                        width: 430px;  /* Set width to match the drawing canvas */
                        height: 10px;
                        margin-top: 10px;
                        color: #0057e7; /* Change progress bar color here */
                        background-color: #eee;
                        border-radius: 3px;
                    }
                    progress::-webkit-progress-bar {
                        background-color: #eee;
                        border-radius: 3px;
                    }
                    progress::-webkit-progress-value {
                        background-color: #0057e7;
                        border-radius: 3px;
                    }
                    #.responses {
                    #margin-top: 20px;
                    #line-height: 1.6;
                    #background-color: white;
                    #padding: 20px;
                    #border-radius: 5px;
                    #box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                #}
                #reflectionContainer {
                    display: none;
                    background-color: #f0f8ff;
                    padding: 10px;
                    border-radius: 5px;
                }
                .active-tool {
                    background-color: black;
                    color: white;
                }

                img {
                    width: 256px;
                    height: 256px;
                    margin: 10px;
                }
                #images img:hover {
                    cursor: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><defs><radialGradient id="grad1" cx="50%" cy="50%" r="50%" fx="50%" fy="50%"><stop offset="0%" style="stop-color:rgb(255,255,255);stop-opacity:0.8" /><stop offset="100%" style="stop-color:rgb(255,255,255);stop-opacity:0.3" /></radialGradient></defs><circle cx="16" cy="16" r="15" fill="url(%23grad1)" stroke="gray" stroke-width="1"/></svg>'), auto;
                }
                    input[type="text"] {
                        width: 600px; /* Increased width for larger input box */
                        height: 40px; /* Optional: Set height for a taller input box */
                        font-size: 18px; /* Increased font size for better readability */
                        padding: 10px; /* Add padding for a better user experience */
                        border: 1px solid #ccc;
                        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
                        border-radius: 4px;
                        transition: box-shadow 0.3s;
                    }
                
                    input[type="text"]:focus {
                        box-shadow: 0px 2px 4px rgba(0,0,0,0.2); 
                        border-radius: 4px;
                    }
                
                .canvas-container {
                    display: flex;
                    align-items: start; /* Align items at the start of the flex container */
                    margin-bottom: 10px;
                    margin-top: 30px; 
                }
            
            canvas {
                    background-color: #f3f4f6;
                    border: 2px solid #cccccc;
                    border-radius: 4px;
                    cursor: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"><circle cx="12" cy="12" r="8" fill="black" fill-opacity="0.4" />stroke="gray" stroke-width="1"/></svg>') 12 12, crosshair;
                }
                .brush {
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    cursor: pointer;
                    display: inline-block;
                    margin: 5px;
                }

                #strokeSizeSlider {
                    width: 200px;
                }

                .tool-button {
                    background-color: white;   /* White background */
                    border: 1.5px solid black;   /* Black border */
                    color: black;              /* Black text */
                    padding: 4px 9px;         /* Padding for better button sizing */
                    cursor: pointer;           /* Pointer cursor on hover */
                    margin-left: 13px;         /* Margin on the left for spacing */
                    border-radius: 4px;        /* Rounded corners */
                }
                
                .spinner {
                    display: inline-block;
                    vertical-align: middle;
                    border: 4px solid rgba(0,0,0,.1);
                    border-radius: 50%;
                    border-left-color: #09f;
                    animation: spin 1s ease infinite;
                    width: 20px;  /* Smaller size */
                    height: 20px; /* Smaller size */
                }

                #loading p {
                    display: inline-block;
                    vertical-align: middle;
                    margin: 0;
                    padding-left: 10px; /* Space between the spinner and the text */
                }

                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }

            </style>


            <script>
                function sendResponse() {
                    const response = document.getElementById('response').value;
                    fetch('/api/question', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({'response': response})
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('question').textContent = data.question;
                        document.querySelector('progress').value = data.progress;
                        document.getElementById('response').value = ''; // Clear the response box
                
                        if (data.progress === 100) {
                            // Show the reflection area when the last question is reached
                            //document.getElementById('reflectionContainer').style.display = 'block';
                            document.getElementById('reflectionContainer').innerHTML = `<div class="responses">${data.responses}</div>`;
                        }
                    })
                    .catch(error => console.error('Error:', error));
                    return false;
                }


                function viewReflection() {
                    document.getElementById('reflectionContainer').scrollIntoView({ behavior: 'smooth' });
                }


                function updateProgressBar() {
                    var currentQuestionNumber = session['question_number'] - 1;  // Assumes this variable is updated correctly from server
                    var progressPercent = currentQuestionNumber * 20;  // Assuming there are 5 questions
                    document.querySelector('progress').value = progressPercent;
                }



                function generateImage(event) {
                    event.preventDefault();  // Prevent the form from submitting traditionally

                    const canvas = document.getElementById('drawingCanvas');
                    const image_data = canvas.toDataURL('image/png');
                    const description = document.getElementById('description').value;

                    document.getElementById('loading').style.display = 'block'; // Show loading indicator

                    fetch('/api/process-drawing', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 'drawing': image_data, 'description': description })
                    })
                    .then(res => res.json())
                    .then(data => {
                        const imagesContainer = document.getElementById('images');
                        data.image_urls.forEach(url => {
                            const img = new Image();
                            img.onload = function() {
                                imagesContainer.insertBefore(img, imagesContainer.firstChild); // Insert new images at the top
                            };
                            img.onclick = function() { replaceCanvas(this.src); }; // use this.src, which is the correct reference
                            img.src = '/proxy?url=' + encodeURIComponent(url); // use url from the forEach loop
                            img.width = 256;
                            img.height = 256;
                        });
                        
                        // Display reappraisal text
                        document.getElementById('reappraisalText').textContent = data.reappraisal_text;
                        document.getElementById('loading').style.display = 'none'; // Hide loading indicator
                    })

                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('loading').style.display = 'none'; // Hide loading indicator if there is an error
                    });

                    return false;
                }


                function replaceCanvas(imgSrc) {
                    const canvas = document.getElementById('drawingCanvas');
                    const ctx = canvas.getContext('2d');
                    const img = new Image();
                    img.crossOrigin = "anonymous";  // Set cross-origin to anonymous
                    img.onload = function() {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    };
                    img.onerror = function() {
                        alert('What do you think about this image?');
                    };
                    img.src = '/proxy?url=' + encodeURIComponent(imgSrc);

                    // After setting the new image, allow the canvas to be used for new drawings or image generations
                    painting = false;  // Reset painting state if needed
                    ctx.beginPath();  // Clear any existing drawing paths
                }

            </script>
        </head>
        <body>
            <div class="container">
                <div class="left">
                <h1>Mind Palette for kids!</h1>
                <div id="question">{{ latest_question }}</div>
                <progress value="{{ progress_value }}" max="100"></progress>  <!-- Progress bar here -->
                <form onsubmit="return sendResponse();">
                    <input type="text" id="response" autocomplete="off" style="width: 430px; margin-top: 15px;" value="" placeholder="Enter your response here..." />
                    <input type="submit" value="Respond" class="button-style" />
                </form>
                <div class="canvas-container ">
                    <canvas id="drawingCanvas" width="500" height="330"></canvas>
                    <button id="backButton" class="tool-button" onclick="undoLastAction()">Back</button>
                </div>
                <div class>
                    <div class="brush" style="background-color: #f44336;" onclick="changeColor('#f44336')"></div>
                    <div class="brush" style="background-color: #ff5800;" onclick="changeColor('#ff5800')"></div>
                    <div class="brush" style="background-color: #faab09;" onclick="changeColor('#faab09')"></div>
                    <div class="brush" style="background-color: #008744;" onclick="changeColor('#008744')"></div>
                    <div class="brush" style="background-color: #0057e7;" onclick="changeColor('#0057e7')"></div>
                    <div class="brush" style="background-color: #a200ff;" onclick="changeColor('#a200ff')"></div>
                    <div class="brush" style="background-color: #ff00c1;" onclick="changeColor('#ff00c1')"></div>
                    <div class="brush" style="background-color: #ffffff; border: 1px solid lightgray;" onclick="changeColor('#ffffff')"></div>
                    <div class="brush" style="background-color: #646765; border: 1px solid lightgray;" onclick="changeColor('#646765')"></div>
                    <div class="brush" style="background-color: black;" onclick="changeColor('black')"></div>
                </div>
                <div style="margin-top: 10px;">
                    Brush size: <input type="range" id="strokeSizeSlider" min="15" max="30" value="2" style="width: 200px;" >
                    <button id="brushButton" class="tool-button" onclick="selectTool('brush')">Brush</button>
                    <button id="eraserButton" class="tool-button" onclick="selectTool('eraser')">Eraser</button>
                </div>



                <script>

                    let currentTool = 'brush'; // Initially set the current tool to brush
                    updateToolButtonStyles();

                    function selectTool(tool) {
                        currentTool = tool;
                        if (tool === 'eraser') {
                            ctx.globalCompositeOperation = 'destination-out';
                            ctx.lineWidth = 20; // Eraser size
                        } else {
                            ctx.globalCompositeOperation = 'source-over';
                            ctx.strokeStyle = currentColor; // Use the selected color
                            ctx.lineWidth = document.getElementById('strokeSizeSlider').value; // Use the slider value
                        }
                        updateToolButtonStyles(); // Update button styles based on the selected tool
                    }

                    function updateToolButtonStyles() {
                        // Remove active class from all buttons
                        document.getElementById('brushButton').classList.remove('active-tool');
                        document.getElementById('eraserButton').classList.remove('active-tool');
                        document.getElementById('backButton').classList.remove('active-tool');

                        // Add active class to the current tool button
                        if (currentTool === 'brush') {
                            document.getElementById('brushButton').classList.add('active-tool');
                        } else if (currentTool === 'eraser') {
                            document.getElementById('eraserButton').classList.add('active-tool');
                        }
                    }

                    function undoLastAction() {
                        if (undoStack.length > 0) {
                            ctx.putImageData(undoStack.pop(), 0, 0);
                            document.getElementById('backButton').classList.add('active-tool');
                            setTimeout(() => {
                                document.getElementById('backButton').classList.remove('active-tool');
                            }, 500); // Remove the active class after 500 ms
                        }
                    }

                    // Bind tool buttons
                    document.getElementById('brushButton').addEventListener('click', () => selectTool('brush'));
                    document.getElementById('eraserButton').addEventListener('click', () => selectTool('eraser'));
                    document.getElementById('backButton').addEventListener('click', undoLastAction);

                    
                    const canvas = document.getElementById('drawingCanvas');
                    const ctx = canvas.getContext('2d');
                    let painting = false;
                    let undoStack = [];  // Stack to keep track of canvas states for undo

                    // Save the current state of the canvas
                    function saveCanvasState() {
                        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                        undoStack.push(imageData);
                    }

                    // Draw on the canvas
                    function draw(event) {
                        if (!painting) return;
                        ctx.lineWidth = document.getElementById('strokeSizeSlider').value;
                        ctx.lineCap = 'round';
                        ctx.lineTo(event.offsetX, event.offsetY);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(event.offsetX, event.offsetY);
                    }

                    // Start painting with mouse down
                    function startPainting(event) {
                        painting = true;
                        draw(event);
                        saveCanvasState();
                    }

                    // Stop painting
                    function stopPainting() {
                        painting = false;
                        ctx.beginPath();
                    }

                    // Undo the last action
                    function undoLastAction() {
                        if (undoStack.length > 0) {
                            const lastState = undoStack.pop();
                            ctx.putImageData(lastState, 0, 0);
                        }
                    }

                    // Set the tool used for drawing
                    function selectTool(tool) {
                        if (tool === 'eraser') {
                            ctx.globalCompositeOperation = 'destination-out';
                            ctx.lineWidth = 20;  // Make the eraser bigger
                        } else {
                            ctx.globalCompositeOperation = 'source-over';
                            ctx.strokeStyle = document.getElementById('currentColor').value;
                        }
                    }

                    // Event listeners for canvas interactions
                    canvas.addEventListener('mousedown', startPainting);
                    canvas.addEventListener('mousemove', draw);
                    canvas.addEventListener('mouseup', stopPainting);
                    canvas.addEventListener('mouseout', stopPainting);

                    // Change color
                    function changeColor(color) {
                        ctx.strokeStyle = color;
                        document.getElementById('currentColor').value = color;
                    }

                    // Buttons for tool selection
                    document.getElementById('brushButton').addEventListener('click', function() { selectTool('brush'); });
                    document.getElementById('eraserButton').addEventListener('click', function() { selectTool('eraser'); });
                    document.getElementById('backButton').addEventListener('click', undoLastAction);

                    // Set initial color
                    let currentColor = '#000000'; // Default black
                    ctx.strokeStyle = currentColor;
                    ctx.lineWidth = 5;
                </script>


                </div>
                <div class="divider"></div>
                <!-- Visual Metaphor section starts here -->
                <div class="right">
                    <h1>Visual Metaphor</h1>
                    <form onsubmit="return generateImage(event);">
                        <label for="description" class="helper-text">
                            I'm here to help you express your emotions. <br> 
                            Please describe what you drew on the canvas! <br>
                        </label><br>
                        <input type="text" id="description" autocomplete="off" style="width: 400px; padding: 5px; margin-top: 10px;" placeholder="Describe your drawing..." />
                        <input type="submit" value="Generate" class="button-style" />
                    </form>
                    <!-- Loading indicator placed right below the form -->
                    <div id="loading" style="display: none; text-align: center;">
                        <div class="spinner"></div>
                        <p>Loading...</p>
                    </div>
                    <div id="images">
                        <!-- Dynamically added images will go here -->
                    </div>
                    <div id="reappraisalText" style="padding: 20px; font-size: 18px; line-height: 1.6; color: black;">
                        <!-- Reappraisal text will appear here -->
                    </div>
                    <input type="button" 
                           value="View Reflections" 
                           class="button-style" 
                           style="background-color: #f3f4f6; color: black;" 
                           onclick="location.href='/reflection'" />
                    <div id="reflectionContainer" style="display: none; margin-top: 20px; padding: 10px; border-radius: 10px; background-color: white; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                        <!-- Reflections will be added dynamically here -->
                    </div>
                </div>


        </body>
    </html>
    """, latest_question=latest_question, progress_value=progress_value)

@app.route('/reflection', methods=['GET'])
def reflection():
    responses = session.get('responses', [])
    formatted_responses = "<br>".join([f"Response {i + 1}: {response}" for i, response in enumerate(responses)])
    return render_template_string("""
    <html>
        <head>
            <title>Your Reflections</title>
            <style>
                body {
                    font-family: 'Helvetica', sans-serif;
                    padding: 20px;
                    background-color: #f0f8ff;
                }
                h1 {
                    color: #333;
                }
                .responses {
                    margin-top: 20px;
                    line-height: 1.6;
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                .button-style {
                    color: white;
                    background-color: black;
                    padding: 5px 10px;
                    cursor: pointer;
                    border: none;
                    margin-left: 10px;
                    border-radius: 4px; 
                }
            </style>
        </head>
        <body>
            <h1>Here is what your kids thought about today.</h1>
            <div class="responses">{{ responses|safe }}</div>
            <button class="button-style" style="margin-top: 20px;" onclick="window.location.href='/'">Restart Session</button>
        </body>
    </html>
    """, responses=formatted_responses)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
