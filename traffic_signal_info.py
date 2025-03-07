
from flask import Flask, request, jsonify
import predict_signal_02 as pd_signal
import openai
from googletrans import Translator
import json
import asyncio


app = Flask(__name__)

model = pd_signal.init()
translator = Translator()


@app.route('/trafficSignal', methods=['GET'])
def explain_traffic_signal():

    image = request.files.get('image')
    language = request.args.get('language')
    region = request.args.get('region')

    if not image:
        return jsonify({"error": "Image not provided"}), 400

    if not language:
        language = 'en'

    if not region:
        region = 'Saskatchewan'

    # Optionally save the image or process it
    # image.save('uploaded_image.jpg')
    prediction = pd_signal.prediction(image, model)

    # Translate text to a target language (e.g., French)
    translated = get_translation(prediction, 'en', language)

    return jsonify({
        "message": "Image and data received successfully",
        "language": language,
        "region": region,
        "image_name": image.filename,
        "result": prediction,
        "translated": translated
    })

async def translate(text, source_lan='en', dest_lan='en'):
    translated = await translator.translate(text, src=source_lan, dest=dest_lan)
    return translated.text

def get_translation(text, source_lan='en', dest_lan='en'):
    return asyncio.run(translate(text, source_lan, dest_lan))

# Endpoint to return the quiz
@app.route('/quiz', methods=['GET'])
def get_quiz():
    language = request.args.get('language')
    module_name = request.args.get('moduleName')

    if not language:
        language = 'en'

    if not module_name:
        module_name = 'traffic signals'

    prompt_quiz = f"""
    Create a JSON response for a road safety quiz on "{module_name}" in {language}.
    Strictly return **only valid JSON**, without explanations or additional text.
    Include:
    1️⃣ `quiz`: A list of 5 multiple-choice questions.
    Each question must include:
       - `question`: The quiz question.
       - `options`: Dictionary format with labeled choices:
         - `a`: Answer choice A
         - `b`: Answer choice B
         - `c`: Answer choice C
         - `d`: Answer choice D
       - `answer`: The correct answer in letter format ("a", "b", "c", or "d").
    Ensure the response is **valid JSON** format.
    """

    response = 'No content'

    client = openai.OpenAI(
        api_key="API_KEY")
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt_quiz}]
        )
        print("API Key is working! Response:")
        print(response)
    except openai.AuthenticationError:
        print("Invalid API key!")
    except openai.OpenAIError as e:
        print("Error:", e)

    quiz_json = json.loads(response.choices[0].message.content)

    return jsonify(quiz_json)



# Endpoint to learn about traffic rules
@app.route('/learn', methods=['GET'])
def learn_traffic():
    language = request.args.get('language')
    module_name = request.args.get('moduleName')

    if not language:
        language = 'en'

    if not module_name:
        module_name = 'traffic signals'

    prompt_learning = f"""
    Generate a JSON response for the "{module_name}" road safety module in {language}.
    Strictly return **only valid JSON**, without explanations or additional text.
    Include:
    1️⃣ `title`: The module title.
    2️⃣ `intro`: A short introduction (2-3 sentences) about the module.
    3️⃣ `learningObjectives`: A list of 5 learning objectives, each with:
       - `objective`: The key learning point.
       - `content`: A short explanation (2-3 sentences).
    Ensure the response is **valid JSON** format.
    """

    response = 'No content'

    client = openai.OpenAI(
        api_key="API_KEY")
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt_learning}]
        )
        print("API Key is working! Response:")
        print(response)
    except openai.AuthenticationError:
        print("Invalid API key!")
    except openai.OpenAIError as e:
        print("Error:", e)

    quiz_json = json.loads(response.choices[0].message.content)

    return jsonify(quiz_json)


@app.route('/', methods=['GET'])
def say_hello():
    response = """
    <h1>Hello World!</h1>
    <h3>Available endpoints are:</h3>
    - <strong>/trafficSignal</strong>    -> to identify a traffic signal (in jpg or jpeg format)<br>
    - <strong>/learn</strong>             -> to get some learning text about traffic<br>
    - <strong>/quiz</strong>              -> to get a test of questions with multiple options<br>
    """
    return response


if __name__ == '__main__':
    app.run(debug=True)



