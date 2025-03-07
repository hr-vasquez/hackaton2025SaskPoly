
from flask import Flask, request, jsonify
import predict_signal_02 as pd_signal
import openai
from googletrans import Translator
import json
import asyncio
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pd_signal.init()
translator = Translator()

API_KEY = "MY_API_KEY"

def encode_image(image_file):
    try:
        # Ensure that image_file is a file-like object and readable
        return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


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


# Endpoint to get traffic signal but using gpt-4
@app.route('/trafficSignal/v2', methods=['POST'])
def explain_traffic_signal_v2():
    image = request.files.get('image')
    user_question = request.args.get('userQuestion')
    language = request.args.get('language')
    region = request.args.get('region')

    if not image and not user_question:
            return jsonify({"error": "Image and question not provided"}), 400

    if not language:
        language = 'en'

    if not region:
        region = 'Saskatchewan'

    if image:
        base64_image = encode_image(image)

        prompt = f"""
        You are an expert in Saskatchewan road signs. Identify the road sign in the image and provide:
        - The name of the road sign.
        - Its meaning.
        - Any driving instructions related to the sign.
        Respond in the language corresponding to this language code: {language}.
        Keep the response short (1-2 sentences).
        """

        client = openai.OpenAI(api_key=API_KEY)

        response = 'No content'
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            print("API Key is working! Response:")
            print(response)
        except openai.AuthenticationError:
            print("Invalid API key!")
        except openai.OpenAIError as e:
            print("Error:", e)

        return response.choices[0].message.content

    elif user_question:
        # API Call for Driving Rules Question
        prompt = f"""
        You are a Saskatchewan driving rules expert. Answer the user's question in their selected language ({language}).
        - Keep the response **short** (1-2 sentences).
        - Ensure answers are **fact-based** and **relevant to Saskatchewan's driving laws**.

        User Question: {user_question}
        """

        client = openai.OpenAI(api_key=API_KEY)

        response = 'No content'
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            print("API Key is working! Response:")
            print(response)
        except openai.AuthenticationError:
            print("Invalid API key!")
        except openai.OpenAIError as e:
            print("Error:", e)

        return response.choices[0].message.content

    else:
        return "Unable to get required image or user question."


# Endpoint to get an answer to a specific query
@app.route('/query', methods=['GET'])
def answer_question():
    user_question = request.args.get('userQuestion')
    language = request.args.get('language')

    if not user_question:
        user_question = 'Do I need to buy winter wheels for winter?'

    if not language:
        language = 'en'

    prompt = f"""
    You are a Saskatchewan driving rules expert. Answer the user's question in the same language they use.
    - Keep the response **short** (1-2 sentences).
    - Ensure answers are **fact-based** and **relevant to Saskatchewan's driving laws**.
    - Ensure to answer in the language {language}

    User Question: {user_question}
    """

    response = 'No content'

    client = openai.OpenAI(
        api_key=API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        print("API Key is working! Response:")
        print(response)
    except openai.AuthenticationError:
        print("Invalid API key!")
    except openai.OpenAIError as e:
        print("Error:", e)

    return response.choices[0].message.content


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
        api_key=API_KEY)
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
        api_key=API_KEY)
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

    learn_json = json.loads(response.choices[0].message.content)

    return jsonify(learn_json)


@app.route('/', methods=['GET'])
def say_hello():
    response = """
    <h1>Hello World!</h1>
    <h3>Available endpoints are:</h3>
    - <strong>/trafficSignal</strong>    -> to identify a traffic signal (in jpg or jpeg format)<br>
    - <strong>/trafficSignal/v2</strong>    -> Same as above but we are using gpt-4 to recognize the image<br>
    - <strong>/query</strong>       -> to answer a specific question that the user has<br>
    - <strong>/learn</strong>             -> to get some learning text about traffic<br>
    - <strong>/quiz</strong>              -> to get a test of questions with multiple options<br>
    """
    return response


if __name__ == '__main__':
    app.run(debug=True)